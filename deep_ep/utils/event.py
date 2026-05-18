# event.py 是 Python API 的异步同步层：它不直接 launch kernel，而是包装 C++ EventHandle。
import torch  # 使用 torch.Tensor 和 torch.cuda.current_stream 相关事件语义。
# Any 用于 with 上下文协议，Optional 表示 event 可为空，Tuple 保存需要随 event 保活的 Tensor。
from typing import Any, Optional, Tuple  # 引入上下文管理器返回值、可选 EventHandle、额外 Tensor tuple 的类型标注。

# noinspection PyUnresolvedReferences
# EventHandle 来自 deep_ep._C 扩展，内部持有 CUDA event 和可能需要 record 的 tensor。
from deep_ep._C import EventHandle  # C++ 扩展暴露的 CUDA event 句柄类型。


# EventOverlap 是所有 async_finish=True API 的返回同步对象；名字强调通信和计算可以重叠。
class EventOverlap:  # Python 侧事件包装器，用于异步通信与计算重叠。
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    中文说明：
        这个类不是普通的 Python Event，而是 DeepEP 异步通信 API 的同步契约。
        dispatch/combine 在通信 stream 上提交 kernel 后，会返回 EventOverlap；
        用户可以先在当前 compute stream 排入一些不依赖通信结果的工作，随后调用
        current_stream_wait 或使用 with 语法，让当前 stream 等待通信 event。
        extra_tensors 的作用是延长相关 Tensor 生命周期，避免 CUDA allocator 在通信
        还未完成时复用底层显存。

    Attributes:
        中文：event 负责 GPU stream 依赖，extra_tensors 只负责 Python 引用生命周期。
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(self,
                 event: Optional[EventHandle] = None,  # C++ 侧 CUDA event 句柄；None 表示同步完成或无需等待。
                 extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:  # 需要随 event 保活的张量集合。
        # 构造函数只保存引用，不会等待 GPU；真正等待发生在 current_stream_wait 或 __exit__。
        """
        Initialize the class.

        Arguments:
            中文：构造时不等待 event，因此创建 EventOverlap 的成本很低。
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        # event 可能为空：例如同步模式 API 仍返回 EventOverlap，但不需要调用者等待。
        self.event = event  # 保存 C++ EventHandle；None 表示当前没有需要等待的事件。

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        # EventHandle 是 C++ 侧记录的 CUDA event；extra_tensors 用来延长相关 tensor 生命周期。
        # 例如 async dispatch 返回后，recv_x 可能还在通信 stream 上被写入，Python 若过早释放中间 tensor，
        # CUDA graph 或 allocator 复用内存会引入数据竞争。把 tensor 绑定到 EventOverlap 可以模拟
        # record_stream 的“等事件完成后再释放”效果。
        # TODO: `extra_tensors` is not longer useful for EPv2, as objects are stored in `self.event`
        # extra_tensors 的 tuple 形状不固定，通常包含输入、输出、layout、统计 tensor 等。
        self.extra_tensors = extra_tensors  # 保存额外张量引用，防止异步期间被释放或复用。

        # A wrapper for `with event_overlap(release_handle=True)`
        # 这个标志只对下一次 with 生效，__exit__ 后会重置为 False。
        self._release_handle_by_call = False  # 记录 __call__ 传入的 release_handle，供 with 退出时使用。

    def current_stream_wait(self, release_handle: bool = False) -> None:
        # 当前 stream 指 torch.cuda.current_stream()；调用者可先切换 stream 再调用本方法。
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.

        中文说明：
            该方法只让“当前 CUDA stream”等待通信 event，不会做全局同步。
            release_handle=True 适合最后一次等待；如果同一个 event 还要被多个 stream
            等待，就不要释放句柄，否则后续 stream 将无法再等待它。
        """
        # 典型用法：recv_x, ..., event = buffer.dispatch(..., async_with_compute_stream=True)
        # 在真正把 recv_x 交给 expert GEMM 前，当前 compute stream 调用 event.current_stream_wait()，
        # 只等待通信完成，而不是全局 cudaDeviceSynchronize。
        # assert 放在这里可以尽早暴露“重复 release 后再次等待”的错误用法。
        assert self.event is not None      # 没有事件时等待没有意义，说明调用方逻辑有误。
        # C++ EventHandle 内部通常调用 cudaStreamWaitEvent，把依赖插入当前 CUDA stream。
        self.event.current_stream_wait()   # 让当前 CUDA stream 等待通信 stream 上记录的 event；不会阻塞 CPU 线程到 GPU 全局空闲。

        # In `self.event`, we also have some V2 APIs storing tensors to record in it,
        # So, after waiting the current stream, those tensors can be released by deleting `self.event`
        # However, you better do it by yourself (to be compatible with multi-stream waits)
        # 多 stream 场景下，只有最后一个等待 event 的 stream 才应该传 release_handle=True。
        if release_handle:                 # 如果用户确认不会再有其它 stream 等待该 event。
            self.event = None              # 释放 C++ EventHandle 和其中保存的 tensor 引用。

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        # __call__ 让对象既能像普通 event 使用，也能写成 event(release_handle=True) 的 with 入口。
        """
        Configures the 'release_handle' behavior for the upcoming context manager usage.
        Usage:
            中文：典型用法是让通信 event 在 with 退出时被当前 stream 等待。
            with event_overlap(release_handle=True):
                ...
        Returns `self` to ensure no new wrapper object is created, keeping the reference count of the underlying event unchanged/managed solely by this instance.

        中文说明：
            __call__ 只记录一次性的 release 策略，然后返回 self，因此
            `with event(release_handle=True):` 不会创建第二个包装对象。
        """
        # 这里只记录配置，不等待 event；等待仍然推迟到 __exit__。
        self._release_handle_by_call = release_handle  # 记录即将进入 with 上下文时的释放策略。
        return self                                    # 返回自身，避免创建新 wrapper 改变引用生命周期。

    def __enter__(self) -> Any:
        # with 入口故意是 no-op，这样用户能在 with 块中把可重叠的计算排入当前 stream。
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        # 中文：下面的计算先入队，退出 with 后再插入等待通信完成的依赖。
        with event_overlap:
            do_something_on_current_stream()
        # 中文：with 退出点相当于在当前 CUDA stream 上插入 wait_event。
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        中文：with 块中的 kernel 会先排入当前 stream；退出 with 时当前 stream 再等待通信 event。

        中文说明：
            进入 with 时刻意不等待，这样 with 内部的计算 kernel 可以先排队；
            等到 __exit__ 时再等待通信 event，从而形成 communication/compute overlap。
        """
        # 返回 self 方便用户在 with 块中访问 event.event 或手动调用其它方法。
        return self  # 进入 with 块时不等待，允许用户在当前 stream 上先排入计算 kernel。

    def __exit__(self,
                 exc_type: Any,  # with 块内异常类型；这里不吞异常，只负责等待 event。
                 exc_val: Any,   # with 块内异常对象。
                 exc_tb: Any) -> None:  # with 块内 traceback。
        # __exit__ 的返回值为 None，表示不吞掉 with 块内异常。
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.

        中文说明：
            无论 with 内部是否抛异常，退出时都会执行这里的等待逻辑；
            这样可以减少异常路径中异步通信仍在写显存而 Python 继续释放对象的风险。
        """
        # 异常路径也等待 event，是为了让 Python 对象生命周期和 GPU 写入生命周期保持一致。
        if self.event is not None:  # 如果仍有事件，退出 with 时让当前 stream 等待它。
            self.current_stream_wait(release_handle=self._release_handle_by_call)  # 按 __call__ 配置决定是否释放句柄。
        # 重置开关后，同一个 EventOverlap 再次进入 with 时默认不释放句柄。
        self._release_handle_by_call = False  # 重置一次性 release 配置，避免影响下一次 with。
