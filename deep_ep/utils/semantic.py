from typing import Any, Optional  # 引入通用类型 Any 和可选类型 Optional，供工具函数标注使用。
import weakref                  # weakref 用于弱引用 self，避免 lru_cache 强引用实例。
import functools                # functools 提供 lru_cache 和 wraps 装饰器。


def value_or(value: Optional[Any], default: Any) -> Any:  # 如果 value 为 None，则返回默认值。
    return default if value is None else value            # 这是 Python 侧处理可选参数的轻量 helper。


def weak_lru(maxsize: Optional[int] = 128, typed: bool = False):  # 为实例方法创建弱引用版本的 lru_cache。
    """LRU Cache decorator that keeps a weak reference to `self`."""  # 装饰器说明：缓存实例方法但不强引用 self。
    def wrapper(func):  # wrapper 接收原始实例方法 func。

        @functools.lru_cache(maxsize, typed)  # 真正缓存的是弱引用对象和其它参数，而不是强引用 self。
        def _func(_self, *args, **kwargs):    # _self 是 weakref.ref(self)，调用时需要先解引用；args/kwargs 是原方法其它参数。
            # functools.lru_cache 如果直接装饰实例方法，会把 self 强引用放入 cache key，
            # 导致 Buffer/ElasticBuffer 这类持有 GPU/NCCL 资源的对象无法释放。这里把 self
            # 包成 weakref，cache 命中时再解引用调用原方法，避免缓存阻止对象析构。
            return func(_self(), *args, **kwargs)  # 解出原 self 后调用原方法；例如 self 已释放时这里会得到 None 并暴露调用错误。

        @functools.wraps(func)               # 保留原函数名、docstring 等元信息。
        def inner(self, *args, **kwargs):    # inner 是最终挂到类方法上的包装函数；self 仍按普通实例方法传入。
            return _func(weakref.ref(self), *args, **kwargs)  # 把 self 转成弱引用后进入缓存函数。

        return inner  # 返回包装后的实例方法。

    return wrapper  # 返回装饰器本身。
