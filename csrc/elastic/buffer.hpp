#pragma once  // ElasticBuffer runtime 是 header-only 绑定实现，防止重复定义。

#include <cuda_runtime.h>       // cudaMallocHost/cudaMemcpyAsync/cudaMemset 等 runtime API。
#include <memory>               // shared_ptr 管理 NCCL context 和 workspace layout。
#include <vector>               // AGRS/PP 等路径使用 vector 保存指针、大小和 Tensor 列表。
#include <pybind11/functional.h> // 允许 std::function hook 返回给 Python。

#include <deep_ep/common/layout.cuh>   // WorkspaceLayout/BufferLayout 定义 GPU window 内存布局。
#include <deep_ep/common/compiled.cuh> // nv_bfloat16、静态断言等公共 CUDA 编译定义。

#include "../kernels/backend/api.cuh"  // NCCLSymmetricMemoryContext 和 cuda_driver flag helper。
#include "../kernels/elastic/api.hpp"  // Elastic JIT kernel launch_* host wrapper。
#include "../utils/event.hpp"          // EventHandle 和 stream_wait，用于跨 stream 同步。
#include "utils.hpp"                   // get_shape/get_data_ptr/get_global_comm_stream helper。

namespace deep_ep::elastic {  // Elastic/V2 C++ runtime 命名空间。

class ElasticBuffer {
    // Buffer registered for both scaleout and scaleup
    // NCCLSymmetricMemoryContext 会注册一整块 symmetric window：
    // [WorkspaceLayout 固定元数据区 | buffer 通信 payload 区]。
    // workspace 用于 barrier、计数、prefix-sum 等控制信息；buffer 用于实际 token payload。
    int64_t num_buffer_bytes;  // payload buffer 字节数，不含 WorkspaceLayout 固定控制区。
    void* buffer;              // payload buffer 起始地址，等于 symmetric window 基址 + WorkspaceLayout::get_num_bytes()。

    // Destructor settings
    bool explicitly_destroy;   // true 表示 Python 必须显式调用 destroy，析构只报警不强制释放。
    bool destroyed = false;    // 防止 destroy 重复执行；NCCL window deregister 不能重复调用。

    // Workspace
    // NOTES: for all workspace, we must keep them as zeros
    // workspace 在 GPU window 内，所有 rank 都能通过 NCCL/Gin 访问；host_workspace 是 pinned
    // + mapped host memory，device kernel 可直接写入小量统计值，CPU 可轮询读取。二者的布局
    // 都由 deep_ep/include/deep_ep/common/layout.cuh::WorkspaceLayout 解释。
    void *workspace;  // GPU symmetric window 内固定 workspace 基址，所有 rank 可远端访问。
    void *host_workspace, *mapped_host_workspace;  // pinned host workspace 及其 device-mapped 指针，用于 CPU/GPU 小量同步。
    std::shared_ptr<layout::WorkspaceLayout> workspace_layout_wo_expert;  // 不含 expert layout 的 workspace 解释器。

    // CUDA streams
    at::cuda::CUDAStream comm_stream;  // 全局 high-priority 通信 stream。

    // Whether to use deterministic algorithms
    bool deterministic;  // 是否使用确定性 dispatch prologue/copy 路径。

    // Whether to use hybrid mode (scale-out with scale-up)
    bool allow_hybrid_mode;  // 是否允许 scaleout RDMA + scaleup NVLink 分层路径。

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;  // combine 是否允许多阶段 reduction 以换取更少传输。

    // Whether to prefer overlapping communication with compute (use more SMs and channels if false)
    bool prefer_overlap_with_compute;  // true 表示更少占用 SM，给专家 GEMM 留资源。

    // Timeout settings
    int num_cpu_timeout_secs;       // CPU 侧等待/轮询超时时间，单位秒。
    int64_t num_gpu_timeout_cycles; // GPU 侧 barrier/轮询超时，单位 clock cycles。

    // NCCL context
    std::shared_ptr<nccl::NCCLSymmetricMemoryContext> nccl_context;  // NCCL communicator/window/Gin device context。

    // Some EP hybrid mode settings
    static constexpr int kNumMaxChannelsPerSM = 8;  // 每个 SM 最多 channel 数，用于 hybrid worst-case buffer 估算。
    static constexpr int kNumMaxSMs = 160;          // 支持的最大 SM 数上界，用于静态最坏 channel 数。
    static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;  // 最大 channel 总数。

    // Some Engram storage settings
    int num_engram_entries = 0, engram_hidden = 0;   // Engram storage 条目数和每条 hidden 维度。
    int64_t num_engram_storage_bytes = 0;            // buffer 前段 storage 字节数，按 32B 对齐。
    int64_t num_engram_recv_bytes = 0;               // buffer 剩余可作为 Engram fetch 输出区的字节数。

    // PP settings
    int prev_rank_idx = 0, next_rank_idx = 0;        // PP ring 中当前 rank 的前驱/后继 rank。
    int64_t num_max_pp_tensor_bytes = 0;             // 单个 PP tensor 槽最大字节数，按 32B 对齐。
    int num_max_pp_inflight_tensors = 0;             // 每个方向最大 inflight tensor 槽数。

    // AGRS session settings
    int64_t num_max_agrs_session_bytes = 0;          // 一个 AGRS session 可使用的最大 buffer 字节数。
    int num_max_agrs_per_session = 0;                // 一个 session 内最多 inflight all_gather 次数。
    int agrs_session_idx = 0;                        // AGRS session 轮次，用于 signal 区分新旧 session。
    bool agrs_in_session = false;                    // 当前是否处于 AGRS session 内。

    // AGRS in-session settings
    int64_t agrs_buffer_offset = 0;                  // 当前 session 内线性分配到的 buffer offset。
    int agrs_buffer_slot_idx = 0;                    // 当前 session 内第几个 all_gather 槽。

public:
    ElasticBuffer(const int& rank_idx, const int& num_ranks,  // 当前 rank 和 EP world size。
                  const int64_t& nccl_comm,  // Python 传入的 ncclComm_t 整数句柄。
                  const int64_t& num_buffer_bytes,  // payload buffer 字节数，不含 workspace。
                  const bool& deterministic,  // 是否使用确定性算法。
                  const bool& allow_hybrid_mode,  // 是否允许 hybrid scaleout/scaleup。
                  const bool& allow_multiple_reduction,  // combine 是否允许多阶段归约。
                  const bool& prefer_overlap_with_compute,  // 是否优先 overlap，少占 SM。
                  const int& sl_idx, const int& num_allocated_qps,  // RDMA service level 和预分配 QP 数。
                  const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,  // CPU/GPU 超时秒数。
                  const bool& explicitly_destroy):  // 是否要求显式 destroy。
        num_buffer_bytes(num_buffer_bytes),  // 初始化 payload buffer 字节数。
        explicitly_destroy(explicitly_destroy),  // 保存析构策略。
        comm_stream(get_global_comm_stream()),  // 取全局 high-priority 通信 stream。
        deterministic(deterministic),  // 保存确定性开关。
        allow_hybrid_mode(allow_hybrid_mode),  // 保存 hybrid 开关。
        allow_multiple_reduction(allow_multiple_reduction),  // 保存多阶段 reduction 开关。
        prefer_overlap_with_compute(prefer_overlap_with_compute) {  // 保存 overlap 偏好。
        // Init NCCL runtime
        // 这里把用户已有的 ncclComm_t 包装成 NCCL Gin symmetric memory context。
        // window 大小 = WorkspaceLayout::get_num_bytes() + num_buffer_bytes，前半部分固定给控制面，
        // 后半部分交给 dispatch/combine/Engram/PP/AGRS 复用。
        static constexpr int kBufferAlignment = 16;  // NCCL window payload 至少 16B 对齐，满足向量化访问基本要求。
        this->nccl_context = std::make_shared<nccl::NCCLSymmetricMemoryContext>(
            nccl_comm, num_ranks, rank_idx,  // communicator、world size、当前 rank。
            layout::WorkspaceLayout::get_num_bytes() + num_buffer_bytes, kBufferAlignment,  // window 总字节数和对齐。
            allow_hybrid_mode, sl_idx, num_allocated_qps);  // hybrid、RDMA SL、QP 配置。

        // Timeout
        // Python 传入的是秒；device 端 barrier/轮询使用 clock cycle，因此需要乘以 GPU clock rate。
        // num_gpu_timeout_cycles 会继续传入 deep_ep/include/deep_ep/common/comm.cuh 的 barrier 模板。
        this->num_cpu_timeout_secs = num_cpu_timeout_secs;  // 保存 CPU 超时秒数。
        this->num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs);  // 先保存 GPU 超时秒数。
        this->num_gpu_timeout_cycles *= jit::device_runtime->get_clock_rate();  // 转成 clock64 周期数。

        // Assign workspaces and buffers
        // mapped_window_ptr 是本 rank 看到的 symmetric window 基址。buffer 从 workspace 后面开始，
        // 所以后续 layout 计算必须始终和 WorkspaceLayout::get_num_bytes() 保持一致。
        workspace = this->nccl_context->mapped_window_ptr;  // workspace 指向 mapped symmetric window 开头。
        workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(
            workspace, nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks, 0);  // 构造不含 expert 的 workspace layout 解释器。
        buffer = static_cast<uint8_t*>(workspace) + layout::WorkspaceLayout::get_num_bytes();  // payload buffer 紧跟 workspace。
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, layout::WorkspaceLayout::get_num_bytes()));  // workspace 初始必须清零。

        // Allocate host workspaces
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));  // 分配 mapped pinned host workspace。
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));  // 获取 device 可访问的 host workspace 指针。
        std::memset(host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());  // host workspace 初始清零，CPU/GPU 轮询协议依赖 0 状态。

        // We should call a barrier at the end
        // 中文：构造函数结束后需要外部 Python 分布式 barrier，确保所有 rank runtime 都初始化完成。
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
    }

    ~ElasticBuffer() noexcept(false) {  // 析构时按策略释放或报警。
        // 如果 Python 侧没有要求显式 destroy，则析构时自动释放资源。显式 destroy 模式
        // 通常用于测试/benchmark，避免对象被 GC 的时机不确定，从而影响 NCCL window 生命周期。
        if (not explicitly_destroy)  // 非显式销毁模式由析构函数释放资源。
            destroy();               // 执行 barrier、host workspace free 和 NCCL context finalize。

        if (not destroyed) {  // 显式销毁模式但用户未调用 destroy。
            // explicit_destroy=True 但用户忘记调用 destroy() 时给出警告。这里不强行 finalize，
            // 因为析构时其它 rank 可能已经退出，collective window deregister 反而可能 hang。
            printf("`destroy()` is not called before DeepEP elastic buffer destruction, which can leak resources.\n");  // 打印资源泄漏警告。
            fflush(stdout);  // 立即刷新，避免进程退出时日志丢失。
        }
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);  // 防止重复销毁 NCCL window 和 host workspace。

        // 释放 symmetric memory 前必须做一次全局 GPU barrier；否则某个 rank 还在访问
        // 本 rank 的 window，ncclCommWindowDeregister/ncclMemFree 会造成远端非法访问。
        barrier(true, true);  // 在通信 stream 上做 GPU+CPU 同步 barrier，确保所有 rank 停止访问 window。

        // Deallocate host workspaces
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));  // 释放 mapped pinned host workspace。

        // Destroy NCCL context
        nccl_context->finalize();  // 注销 NCCL window、释放 symmetric memory 和 device communicator。

        // Cannot use anymore
        destroyed = true;  // 标记 runtime 已销毁，后续析构不再报警/重复释放。
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;  // 返回通信 stream，pybind 会暴露给 Python 包装成 torch.cuda.Stream。
    }

    std::tuple<int, int> get_physical_domain_size() const {
        return {nccl_context->num_rdma_ranks, nccl_context->num_nvl_ranks};  // 返回物理 RDMA/NVLink 域大小。
    }

    std::tuple<int, int> get_logical_domain_size() const {
        return {nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks};  // 返回逻辑 scaleout/scaleup 域大小。
    }

    // ReSharper disable once CppMemberFunctionMayBeStatic
    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync) const {
        const auto compute_stream = at::cuda::getCurrentCUDAStream();  // 记录进入 barrier 时用户当前 compute stream。
        const auto stream = use_comm_stream ? comm_stream : compute_stream;  // 根据参数选择 barrier kernel 所在 stream。
        // use_comm_stream=true 时，先让通信 stream 等当前 compute stream，保证之前排队的
        // 用户 kernel 写入的数据对 barrier 后的通信 kernel 可见。
        if (use_comm_stream)  // barrier 走通信 stream 时需要先接上 compute stream 之前的工作。
            stream_wait(comm_stream, compute_stream);  // comm_stream 等 compute_stream 当前进度。

        // Wait all streams to finish on this GPU
        if (with_cpu_sync)  // CPU 同步模式用于初始化/销毁等需要强顺序的场景。
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());  // 等本 GPU 所有 stream 已提交工作完成。

        // Launch GPU barrier
        launch_barrier(nccl_context->dev_comm, nccl_context->window,  // 传入 NCCL device communicator 和 symmetric window。
                       workspace,  // barrier 使用的 workspace 信号区。
                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,  // 当前 rank 的逻辑坐标。
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,  // 逻辑域大小。
                       num_gpu_timeout_cycles,  // GPU 轮询超时 cycles。
                       nccl_context->is_scaleup_nvlink,  // scaleup 是否在 NVLink 域内，影响 barrier 路径。
                       stream);  // barrier kernel 所在 stream。

        // Let CPU wait
        if (with_cpu_sync)  // 调用方要求 CPU 等 barrier 完成。
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());  // 等 barrier kernel 和相关通信完成。

        // barrier 放在 comm stream 上时，最后把 compute stream 接回 comm stream；
        // 对 Python 用户来说，barrier() 返回后当前 stream 后续 kernel 不会越过 barrier。
        if (use_comm_stream)  // barrier 放在 comm_stream 时，需要让 compute_stream 看到 barrier 完成。
            stream_wait(compute_stream, comm_stream);  // compute_stream 等 comm_stream 的 barrier event。
    }

    void engram_write(const torch::Tensor& storage) {
        // Engram storage 复用 EP buffer 的前半段，因此写入前先等待之前所有 fetch 完成，
        // 避免还在读旧 storage 的 GIN get 与新的 cudaMemcpyAsync 发生数据竞争。
        barrier(false, true);  // 在当前 compute stream 上做强 barrier，确保旧 Engram 访问完成。

        const auto compute_stream = at::cuda::getCurrentCUDAStream();  // Engram 写入使用用户当前 stream。

        // Check storage
        const auto [num_entries, hidden] = get_shape<2>(storage);  // storage 形状必须是 [num_entries, hidden]。
        EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16);  // 当前 Engram storage 只支持 BF16。
        EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());  // storage 必须是 CUDA contiguous tensor。
        num_engram_entries = num_entries, engram_hidden = hidden;  // 保存 storage 维度，fetch 时还原输出 shape。

        // storage 被拷到 buffer 起始处；随后把 storage 占用字节向 32B 对齐，
        // 对齐后的尾部作为 fetch 输出区。布局：
        //   buffer[0:num_engram_storage_bytes) = storage [num_entries, hidden]
        //   buffer[num_engram_storage_bytes:] = fetched [num_tokens, hidden]
        EP_HOST_ASSERT(storage.nbytes() <= num_buffer_bytes);  // storage 不能超过 payload buffer 总容量。
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            buffer, storage.data_ptr(), storage.nbytes(),  // 从用户 storage 拷到 symmetric payload buffer 起始处。
            cudaMemcpyDeviceToDevice, compute_stream));  // GPU 到 GPU 异步拷贝，排在 compute_stream 上。
        num_engram_storage_bytes = math::align<int64_t>(storage.nbytes(), 32);  // storage 区按 32B 向上对齐。
        num_engram_recv_bytes = num_buffer_bytes - num_engram_storage_bytes;  // 剩余区域用于 fetch 输出。

        // 第二次 barrier 确保所有 rank 都已完成本地 storage 写入。远端 rank 的
        // engram_fetch 会按 global index 直接读取这些 storage 区域。
        barrier(false, true);  // 等所有 rank 完成本地 storage 写入后再允许远端 fetch。
    }

    std::function<torch::Tensor()> engram_fetch(const torch::Tensor& indices, int num_qps) const {
        const auto [num_tokens] = get_shape<1>(indices);  // indices 形状 [num_tokens]。
        // indices 形状 [num_tokens]，元素必须是 int32 global entry id。
        // 输出 fetched 的容量必须能放下 num_tokens * hidden 个 BF16。
        EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);  // global entry id 使用 int32。
        EP_HOST_ASSERT(indices.is_cuda() and indices.is_contiguous());  // indices 必须在 GPU 上且连续。
        EP_HOST_ASSERT(num_tokens * engram_hidden * sizeof(nv_bfloat16) <= num_engram_recv_bytes);  // 输出区必须容纳 [num_tokens,hidden]。

        // num_qps=0 表示使用初始化 NCCL device communicator 时分配的所有 QP。
        // QP 越多并发越高，但也会占用更多 SM/block 和 NIC 队列资源。
        if (num_qps == 0)  // 0 表示自动使用所有已分配 QP。
            num_qps = nccl_context->num_allocated_qps;  // 取 NCCL context 中的 QP 上限。

        // fetched 直接用 from_blob 包装 buffer 的接收区，不额外分配 Tensor storage。
        // 这意味着下一次 engram_fetch/engram_write 可能覆盖它；用户应在 hook() 后及时消费。
        EP_HOST_ASSERT(num_engram_entries > 0);  // 必须先调用 engram_write 初始化 storage。
        const auto fetched = torch::from_blob(
            math::advance_ptr(buffer, num_engram_storage_bytes),  // fetch 输出区起始地址。
            {num_tokens, engram_hidden},  // 输出形状 [num_tokens, hidden]。
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)  // 输出 dtype/device。
        );  // from_blob 不拥有内存，底层存储由 ElasticBuffer payload buffer 管理。

        // last_gin_requests 的逻辑形状是 [num_qps, num_ranks]，这里用 byte tensor
        // 承载 ncclGinRequest_t 的原始字节，保证 PyTorch 管理其生命周期。
        const auto last_gin_requests = torch::empty(
            {nccl_context->num_ranks * num_qps, sizeof(ncclGinRequest_t)},  // 每个 rank/QP 一个 request 结构的字节存储。
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)  // 用 uint8 tensor 保活原始 request bytes。
        );  // last_gin_requests 会被 lambda 捕获，确保 wait 前不释放。

        // fetch kernel 只提交/flush RDMA get，并不等待全部数据完成；返回的 lambda
        // 会在用户需要数据时再 launch wait kernel，从而允许和后续计算重叠。
        launch_engram_fetch(  // 启动 RDMA/Gin fetch issue kernel。
            nccl_context->dev_comm, nccl_context->window,  // NCCL device communicator 和 window。
            buffer,  // Engram storage 所在 symmetric payload buffer。
            fetched.data_ptr(),  // 本 rank fetch 输出地址。
            indices.data_ptr<int>(),  // global entry id 列表。
            static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),  // 保存未完成 Gin request 的数组。
            num_engram_entries, engram_hidden,  // storage 条目数和 hidden。
            num_tokens,  // 本次 fetch 的 token/entry 数。
            nccl_context->num_ranks, num_qps,  // world size 和 QP 数。
            at::cuda::getCurrentCUDAStream()  // 当前 stream 上提交 issue kernel。
        );  // issue kernel 返回后 request 可能仍未完成，需要 hook 等待。

        return [=, this]() {
            // hook 必须在同一逻辑使用点调用：它在当前 CUDA stream 上等待所有
            // last_gin_requests 完成，然后返回 fetched tensor。
            launch_engram_fetch_wait(  // 启动 wait kernel 等待所有 Gin get 完成。
                static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),  // issue 阶段记录的 request 数组。
                nccl_context->dev_comm,  // NCCL device communicator。
                nccl_context->window,  // NCCL symmetric window。
                nccl_context->num_ranks, num_qps,  // request 数组逻辑维度。
                at::cuda::getCurrentCUDAStream()  // 在调用 hook 时的当前 stream 上等待。
            );  // wait 完成后 fetched 数据可安全读取。
            return fetched;  // 返回 from_blob 输出 tensor。
        };  // 返回延迟等待 hook。
    }

    void pp_set_config(const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors) {
        // PP buffer 重新配置会改变环形槽容量和 offset，因此必须先 flush 之前的 send/recv。
        barrier(false, true);  // 在当前 stream 上做强同步，避免旧 PP 操作仍访问旧槽布局。

        EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);  // 槽大小和 inflight 数必须为正。
        // buffer 按 4 份切分：prev/next 两个方向，各有 send staging 与 recv landing 两段。
        // 每段包含 num_max_inflight_tensors 个槽位。
        EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <= num_buffer_bytes);  // 四段 PP buffer 总大小不能超过 payload buffer。
        this->prev_rank_idx = (nccl_context->rank_idx + nccl_context->num_ranks - 1) % nccl_context->num_ranks;  // ring 前驱 rank。
        this->next_rank_idx = (nccl_context->rank_idx + 1) % nccl_context->num_ranks;  // ring 后继 rank。
        this->num_max_pp_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, 32);  // 单槽按 32B 对齐。
        this->num_max_pp_inflight_tensors = num_max_inflight_tensors;  // 保存每方向 inflight 槽数。
    }

    void pp_send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
        // PP send 只允许发给相邻 rank，形成双向 ring。这样 device kernel 只需要维护
        // prev/next 两个方向的槽位 offset，而不需要 num_ranks^2 的连接矩阵。
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);  // 必须先 pp_set_config。
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);  // 输入必须是 CUDA contiguous 且不超过槽容量。
        EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or dst_rank_idx == next_rank_idx);  // 只能发送给 ring 相邻 rank。

        launch_pp_send(  // 启动 PP send JIT kernel。
            nccl_context->dev_comm, nccl_context->window,  // NCCL device communicator 和 symmetric window。
            x.data_ptr(), x.nbytes(),  // 源 tensor 指针和实际发送字节数。
            buffer, workspace,  // payload buffer 和同步 workspace。
            nccl_context->rank_idx, dst_rank_idx, nccl_context->num_ranks,  // 当前 rank、目标 rank、world size。
            num_max_pp_tensor_bytes,  // 单个 PP 槽容量。
            num_max_pp_inflight_tensors,  // 每方向 inflight 槽数。
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,  // SM 数，0 表示使用全卡 SM。
            num_gpu_timeout_cycles,  // signal 等待超时 cycles。
            jit::device_runtime->get_num_smem_bytes(),  // 可用动态 shared memory 上限。
            at::cuda::getCurrentCUDAStream()  // 在当前 stream launch。
        );  // send kernel 会写目标 rank ring buffer 并设置 signal。
    }

    void pp_recv(const torch::Tensor& x, const int& src_rank_idx, const int& num_sms) const {
        // recv 的 x 是用户提供的输出 tensor，kernel 会从 symmetric buffer 的 landing
        // 槽拷入该 tensor。x.nbytes() 可以小于最大槽容量，但必须 contiguous。
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);  // 必须先 pp_set_config。
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);  // 输出 tensor 必须是 CUDA contiguous 且容量足够。
        EP_HOST_ASSERT(src_rank_idx == prev_rank_idx or src_rank_idx == next_rank_idx);  // 只能从 ring 相邻 rank 接收。

        launch_pp_recv(  // 启动 PP recv JIT kernel。
            nccl_context->dev_comm, nccl_context->window,  // NCCL device communicator 和 symmetric window。
            x.data_ptr(), x.nbytes(),  // 目标输出 tensor 指针和实际接收字节数。
            buffer, workspace,  // payload buffer 和同步 workspace。
            nccl_context->rank_idx, src_rank_idx, nccl_context->num_ranks,  // 当前 rank、源 rank、world size。
            num_max_pp_tensor_bytes,  // 单个 PP 槽容量。
            num_max_pp_inflight_tensors,  // 每方向 inflight 槽数。
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,  // SM 数，0 表示使用全卡 SM。
            num_gpu_timeout_cycles,  // signal 等待超时 cycles。
            jit::device_runtime->get_num_smem_bytes(),  // 可用动态 shared memory 上限。
            at::cuda::getCurrentCUDAStream()  // 在当前 stream launch。
        );  // recv kernel 会等待 signal 并把 ring 槽数据拷入 x。
    }

    void agrs_set_config(const int64_t& num_max_session_bytes,
                         const int& new_num_max_agrs_per_session) {
        // AGRS 只支持单节点 NVLink ranks，因为实现依赖 NCCL LSA symmetric pointer
        // 直接 cudaMemcpyBatch 到 peer window。配置变更前先做 barrier 清空旧 session。
        barrier(true, true);  // 配置变更前做全局强同步，避免旧 all_gather 还在访问 buffer。

        EP_HOST_ASSERT(nccl_context->num_ranks > 1);  // AGRS 至少需要两个 rank 才有意义。
        EP_HOST_ASSERT(num_max_session_bytes > 0 and new_num_max_agrs_per_session > 0);  // session 字节数和次数上限必须为正。
        EP_HOST_ASSERT(num_max_session_bytes <= num_buffer_bytes);  // session 最大字节数不能超过 payload buffer。
        EP_HOST_ASSERT(new_num_max_agrs_per_session <= layout::WorkspaceLayout::kNumMaxInflightAGRS);  // inflight 次数不能超过 workspace signal 槽。
        EP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks);  // AGRS 当前仅支持全 rank 在同一 NVLink 域。
        this->num_max_agrs_session_bytes = math::align<int64_t>(num_max_session_bytes, 32);  // session 字节数按 32B 对齐。
        this->num_max_agrs_per_session = new_num_max_agrs_per_session;  // 保存 session 内最大 all_gather 次数。
    }

    void create_agrs_session() {
        // session 内多个 all_gather 共享同一段 buffer，并通过 agrs_buffer_offset 线性分配。
        // session_idx 单调递增，用作 peer 间 signal 的轮次编号，避免旧 signal 被误认为新完成。
        EP_HOST_ASSERT(not agrs_in_session);  // 不允许嵌套 session。
        agrs_in_session = true;  // 标记 session 开始。
        agrs_buffer_offset = 0;  // 新 session 从 buffer 起始处重新分配。
        agrs_buffer_slot_idx = 0;  // 新 session 的 all_gather slot 从 0 开始。
        agrs_session_idx += 1;  // 增加 session 轮次，用于 signal value。
    }

    void destroy_agrs_session() {
        // 结束 session 时需要通知所有 peer：本 rank 不会再写本轮 AGRS buffer；
        // 同时等待所有 peer 的同轮 signal，确保 buffer 可以安全复用。
        EP_HOST_ASSERT(agrs_in_session);  // 必须处于 session 内才能销毁。
        agrs_in_session = false;  // 先标记退出 session，避免后续误提交新 all_gather。

        // Wait compute stream
        stream_wait(comm_stream, at::cuda::getCurrentCUDAStream());  // comm_stream 等当前 compute stream，确保本 rank 写入已排队。

        // write_ptrs 写到远端的 agrs_session_signal[rank]，wait_ptrs 读取本地看到的
        // peer signal。self rank 不需要写/等，因为同 stream 顺序已经保证本地完成。
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);  // 写给其它 rank 的 session 完成 signal 地址。
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);  // 本 rank 等待其它 rank signal 的地址。
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {  // 遍历除自己外的所有 peer。
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;  // 环形顺序枚举 peer rank。
            write_ptrs[i] = static_cast<int*>(  // 第 i 个 peer 上属于本 rank 的 session_signal 地址。
                nccl_context->get_sym_ptr(workspace_layout_wo_expert->get_agrs_session_signal_ptr(nccl_context->rank_idx), dst_rank_idx));  // 转成 peer symmetric ptr。
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_session_signal_ptr(dst_rank_idx);  // 本地等待 peer rank 写回的 signal。
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, agrs_session_idx);  // 在 comm_stream 上批量写本轮 signal 并等待 peer。
    }

    std::vector<torch::Tensor> agrs_get_inplace_tensor(const std::vector<int64_t>& num_bytes_list) const {
        // inplace tensor 直接指向 buffer 中属于本 rank 的切片，布局为
        // [tensor0_rank0, tensor0_rank1, ..., tensor1_rank0, tensor1_rank1, ...]。
        // 用户把输入写到本 rank 切片后，all_gather 只需把这些切片复制到其它 peer。
        EP_HOST_ASSERT(num_bytes_list.size() >= 1);  // 至少需要申请一个 inplace tensor。
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);  // 必须先配置并进入 session。

        std::vector<torch::Tensor> out;  // 输出 byte tensor view 列表。
        out.reserve(num_bytes_list.size());  // 预留 vector 容量，避免 push_back 重新分配。
        int64_t offset = agrs_buffer_offset;  // 从当前 session offset 开始试算，不立即提交到成员变量。
        for (const auto& num_bytes: num_bytes_list) {  // 遍历每个用户请求的 tensor 字节数。
            // 每个 tensor 的所有 rank 切片连续摆放，因此一个 tensor 消耗
            // num_bytes * num_ranks 字节。要求 32B 对齐以匹配 batch memcpy/TMA 对齐。
            EP_HOST_ASSERT(num_bytes % 32 == 0);  // AGRS batch copy 要求每个切片 32B 对齐。
            EP_HOST_ASSERT(offset + num_bytes * nccl_context->num_ranks <= num_max_agrs_session_bytes and
                           agrs_buffer_slot_idx < num_max_agrs_per_session and
                           "Not enough session buffer size. Did you forget to flush session?");  // 检查 session 剩余容量和 slot 数。
            out.push_back(torch::from_blob(math::advance_ptr(buffer, offset + num_bytes * nccl_context->rank_idx),  // 本 rank 切片地址。
                          {num_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)));  // 先按 byte tensor 暴露给 Python。
            offset += num_bytes * nccl_context->num_ranks;  // 跳过该 tensor 的所有 rank 切片。
        }
        return out;  // 返回本 rank 可直接写入的 byte views；真正提交 offset 在 all_gather 中发生。
    }

    std::pair<std::vector<torch::Tensor>, std::function<void()>>
    all_gather(const std::vector<torch::Tensor>& tensors) {
        const int num_tensors = tensors.size();  // 本次 all_gather 的 tensor 个数。
        // all_gather 必须在 session 内调用；session 限制了最大总字节和最大 inflight 次数，
        // 防止用户不断提交 handle 而没有等待，覆盖仍在传输的数据。
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);  // 必须已配置且处于 session 内。
        EP_HOST_ASSERT(num_tensors >= 1);  // 至少 gather 一个 tensor。

        int num_copies = 0;  // 需要提交给 cudaMemcpyBatchAsync 的 copy 数。
        std::vector<int64_t> offset(num_tensors);  // 每个 tensor 在 session buffer 中的起始 offset。
        for (int i = 0; i < num_tensors; ++ i) {  // 遍历输入 tensor。
            const auto& x = tensors[i];  // 当前输入 tensor。
            // 如果 x 已经是 agrs_get_inplace_tensor 返回的本 rank 切片，则本地 rank
            // 不需要再拷贝一次，所以 num_copies 少 1；否则需要把普通 tensor 也复制到本地槽。
            EP_HOST_ASSERT(x.is_contiguous());  // 输入 tensor 必须 contiguous，batch copy 按连续字节搬运。
            EP_HOST_ASSERT(x.is_cuda() and x.nbytes() % 32 == 0);  // 输入必须在 GPU 上且字节数 32B 对齐。

            const auto x_offset = math::ptr_diff(x.data_ptr(), buffer);  // 判断输入是否已经位于 AGRS payload buffer 内。
            const bool is_inplace = 0 <= x_offset and x_offset < num_max_agrs_session_bytes;  // 输入是否来自 agrs_get_inplace_tensor。
            offset[i] = agrs_buffer_offset;  // 记录该 tensor 的 session buffer 起点。
            num_copies += nccl_context->num_ranks - is_inplace;  // inplace 时本地槽无需 copy，否则每个 rank 都需要一份 copy。
            agrs_buffer_offset += x.nbytes() * nccl_context->num_ranks;  // 为该 tensor 的所有 rank 切片推进 offset。
            EP_HOST_ASSERT(not is_inplace or x.data_ptr() == math::advance_ptr(buffer, offset[i] + x.nbytes() * nccl_context->rank_idx));  // inplace 输入必须正好位于本 rank 切片。
        }
        EP_HOST_ASSERT(agrs_buffer_offset <= num_max_agrs_session_bytes and
                       agrs_buffer_slot_idx < num_max_agrs_per_session and
                       "Not enough session buffer size. Did you forget to flush session?");  // 检查 session 容量和 slot 上限。

        // Wait compute stream
        const auto compute_stream = at::cuda::getCurrentCUDAStream();  // 记录当前 compute stream。
        stream_wait(comm_stream, compute_stream);  // comm_stream 等输入 tensor 写入完成。

        // 构造 cudaMemcpyBatchAsync 的三元组：src 是本 rank 的输入切片，dst 是目标 rank
        // symmetric window 中属于本 rank 的槽位。对每个目标 rank、每个 tensor 各一次 copy。
        std::vector<size_t> sizes(num_copies);  // 每个 copy 的字节数。
        std::vector<void*> dst_ptrs(num_copies), src_ptrs(num_copies);  // batch copy 的目标/源指针数组。
        int count = 0;  // 当前填充到第几个 copy descriptor。
        for (int i = 0; i < nccl_context->num_ranks; ++ i) {  // 遍历所有目标 rank，包括自己。
            for (int j = 0; j < num_tensors; ++ j) {  // 遍历本次 session 中的所有 tensor。
                const auto& x = tensors[j];  // 当前输入 tensor。
                const auto dst_rank_idx = (nccl_context->rank_idx + i) % nccl_context->num_ranks;  // 环形顺序枚举目标 rank。
                void* src_ptr = x.data_ptr();  // 源指针是本 rank 输入 tensor。
                void* dst_ptr =  // 目标指针是 dst rank symmetric buffer 中“本 rank 切片”位置。
                    nccl_context->get_sym_ptr(math::advance_ptr(buffer, offset[j] + x.nbytes() * nccl_context->rank_idx), dst_rank_idx);  // 转成远端对称地址。
                if (src_ptr != dst_ptr) {  // inplace 本地槽可能源/目标相同，此时跳过 copy。
                    src_ptrs[count] = src_ptr;  // 保存第 count 个 copy 的源地址。
                    dst_ptrs[count] = dst_ptr;  // 保存第 count 个 copy 的目标地址。
                    sizes[count] = x.nbytes();  // 保存第 count 个 copy 的字节数。
                    count += 1;  // 推进 copy descriptor 数。
                }
            }
        }
        cudaMemcpyAttributes attrs = {  // batch memcpy 属性。
            .srcAccessOrder = cudaMemcpySrcAccessOrderStream,  // 源数据访问顺序跟随 stream，保证和前序写入有序。
            .flags = cudaMemcpyFlagPreferOverlapWithCompute  // 提示 CUDA 尽量与计算重叠。
        };  // attrs 传给 cudaMemcpyBatchAsync。
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, comm_stream));  // CUDA 13+ API 签名。
#else
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, nullptr, comm_stream));  // CUDA 12.x API 签名。
#endif

        // 数据 copy 入队后，写远端 recv_signal[slot, rank] 并等待所有 peer 写回本地
        // recv_signal[slot, peer]。slot_idx 区分同一 session 内多个 inflight all_gather。
        const int current_session = agrs_session_idx;  // 捕获当前 session 轮次。
        const int slot_idx = agrs_buffer_slot_idx;  // 本次 all_gather 使用的 inflight slot。
        agrs_buffer_slot_idx += 1;  // 为下一次 all_gather 推进 slot。
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);  // 写给 peer 的 recv_signal 地址。
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);  // 本地等待 peer 写回的 recv_signal 地址。
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {  // 遍历除自身外的 peer。
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;  // 目标 peer rank。
            write_ptrs[i] = nccl_context->get_sym_ptr(  // 第 i 个 peer 上 slot_idx 中属于本 rank 的 signal。
                workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, nccl_context->rank_idx), dst_rank_idx);  // 转成远端 symmetric ptr。
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, dst_rank_idx);  // 本地等待 peer 的 signal。
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, current_session);  // 写本 rank 完成并等待 peer 完成。

        // 输出 tensor 不拷贝数据，只是从 buffer offset 创建 view，形状在原 shape 前
        // 增加 num_ranks 维。例如输入 [32,64,2048] 输出 [num_ranks,32,64,2048]。
        std::vector<torch::Tensor> out(num_tensors);  // 保存每个输入对应的 gathered view。
        for (int i = 0; i < num_tensors; ++ i) {  // 遍历所有输出。
            auto shape = tensors[i].sizes().vec();  // 复制原 tensor shape。
            shape.insert(shape.begin(), nccl_context->num_ranks);  // 在最前面插入 rank 维。
            out[i] = torch::from_blob(math::advance_ptr(buffer, offset[i]), shape, tensors[i].options());  // 从 session buffer 创建 gathered view。
        }

        // handle 等待 comm_stream event。用户拿到 out 后必须调用 handle() 或在
        // Python wrapper 中等待，否则读取 out 可能早于 peer copy 完成。
        const auto event = EventHandle(comm_stream);  // 在 comm_stream 上记录 all_gather 完成事件。
        auto handle = [=, this]() {  // 返回给 Python 的等待 hook。
            EP_HOST_ASSERT(compute_stream == at::cuda::getCurrentCUDAStream());  // hook 应在同一个 compute stream 语境调用。
            EP_HOST_ASSERT(agrs_in_session and current_session == this->agrs_session_idx);  // hook 必须在同一 session 内等待。
            stream_wait(compute_stream, event);  // 让 compute stream 等 all_gather copy/signal 完成。
        };  // hook 不返回值，只建立 stream 依赖。
        return {std::move(out), std::move(handle)};  // 返回 gathered views 和等待 hook。
    }

    torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,  // 可选前序事件，表示通信 stream 要等的外部工作。
                                                    const bool& allocate_on_comm_stream,  // 是否把后续 Tensor 分配归属切到 comm_stream。
                                                    const bool& async_with_compute_stream) const {  // 是否异步返回给 compute stream。
        // dispatch/combine 的输出 tensor 可能由通信 stream 写入。若 async_with_compute_stream
        // 开启，最好在 comm_stream 上分配这些 tensor，使 PyTorch caching allocator
        // 记录正确的 stream 依赖；否则 compute stream 可能过早复用同一块显存。
        const auto compute_stream = at::cuda::getCurrentCUDAStream();  // 记录调用方进入 API 时的当前 compute stream。
        if (allocate_on_comm_stream)  // 如果输出 tensor 将由 comm_stream 写入，先切换当前 stream。
            at::cuda::setCurrentCUDAStream(comm_stream);  // 让后续 torch::empty 等分配记录到 comm_stream。

        // Assertion for safety
        // `previous_event` implicitly means the overlapping computation kernels are launched first,
        // in order not to use the memory on the compute stream, we must allocate on the communication stream.
        // 中文：否则 compute stream 分配的 tensor 可能在通信 stream 尚未写完时被 allocator 误判为可复用。
        // If you launch the communication kernels firstly, then `previous_event` must be unnecessary.
        if (previous_event.has_value())  // 传 previous_event 表示用户已经在其它 stream 安排前序计算。
            EP_HOST_ASSERT(allocate_on_comm_stream);  // 这种 overlap 模式要求输出分配也在 comm_stream 上。

        // previous_event 用于通信和外部计算重叠：如果提供，comm_stream 等这个 event；
        // 否则 comm_stream 等当前 compute stream，保持默认串行语义。
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());  // comm_stream 等外部 event。
        } else {
            stream_wait(comm_stream, compute_stream);  // 默认让 comm_stream 等当前 compute stream。
        }
        return compute_stream;  // 返回原 compute stream，epilogue 需要恢复和建立等待关系。
    }

    void stream_control_before_epilogue(const std::optional<EventHandle>& previous_event_before_epilogue) const {
        // 有些调用希望主通信 kernel 与后处理 epilogue 之间插入额外依赖，例如等待专家
        // GEMM 产出 recv_x。这个 event 只影响 epilogue 前的 comm_stream。
        if (previous_event_before_epilogue.has_value())  // 若用户提供 epilogue 前事件。
            stream_wait(comm_stream, previous_event_before_epilogue.value());  // comm_stream 在 epilogue 前等待该事件。
    }

    std::optional<EventHandle> stream_control_epilogue(const std::vector<std::optional<torch::Tensor>>& tensors,  // 需要保活或 record_stream 的 tensor。
                                                       const at::cuda::CUDAStream& compute_stream,  // prologue 记录的原 compute stream。
                                                       const bool& allocate_on_comm_stream,  // 是否需要恢复 current stream。
                                                       const bool& async_with_compute_stream) const {  // 是否异步返回事件。
        // async 模式下返回 EventHandle，让 Python 用户显式等待；同步模式下直接让
        // compute_stream 等 comm_stream，函数返回时输出 tensor 已可在当前 stream 使用。
        std::optional<EventHandle> event;  // 同步模式返回 nullopt，异步模式返回 comm_stream 完成事件。
        if (async_with_compute_stream) {  // 异步模式不让 compute stream 立即等待 comm_stream。
            event = EventHandle(comm_stream);  // 在 comm_stream 上记录完成事件，交给 Python EventOverlap。

            // NOTES: this environment only applies to V2 APIs
            if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {  // 特殊模式：不调用 PyTorch record_stream。
                event->tensors_to_record = tensors;  // 通过 EventHandle 持有 tensor 引用，避免 allocator 过早复用。
            } else {
                for (auto& t: tensors) if (t.has_value()) {  // 遍历所有非空 tensor。
                    t->record_stream(compute_stream);  // 告诉 caching allocator compute stream 可能使用该 tensor。
                    t->record_stream(comm_stream);  // 告诉 caching allocator comm_stream 正在写/读该 tensor。
                }
            }
        } else {
            stream_wait(compute_stream, comm_stream);  // 同步模式让 compute stream 等通信完成。
        }

        // prologue 可能把 current CUDA stream 切到 comm_stream；这里恢复进入函数前的 compute stream。
        if (allocate_on_comm_stream)  // prologue 改过 current stream 时需要恢复。
            at::cuda::setCurrentCUDAStream(compute_stream);  // 恢复用户原 compute stream。

        // The CUDA event marking the finishing
        return event;  // 返回可选完成事件。
    }

    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,  // 每 rank 最多 token 数。
                                            const int& hidden, const int& num_sf_packs, const int& num_topk,  // token hidden、scale pack 数和 top-k 数。
                                            const int& elem_size,  // payload 元素字节数，BF16=2，FP8=1。
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,  // 逻辑拓扑维度。
                                            const bool& is_scaleup_nvlink) {  // scaleup 域是否等于 NVLink 域。
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;  // 逻辑总 rank 数。
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);  // 单 token 在 dispatch buffer 中的结构。

        if (num_scaleout_ranks == 1) {  // direct 模式，没有跨 scaleout 转发。
            // Direct dispatch buffer：
            // send_buffer 只在非 NVLink scale-up 时需要 staging；NVLink 可直接写 peer recv_buffer。
            // recv_buffer 第一维是来源 rank，第二维是该 rank 最多发送的 token。
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);  // 非 NVLink scaleup 才需要单段 send staging。
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_ranks, num_max_tokens_per_rank);  // 接收区按来源 rank 切分。
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();  // direct dispatch 总字节数。
        } else {
            // Hybrid dispatch 需要三段 buffer：节点内最终接收区、跨 scaleout 发送区、
            // 跨 scaleout 接收区。最后一段 token 数近似为 max_tokens + max_channels，
            // 预留 channel tail 对齐和最坏 channel 切分。
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);  // 最终 scaleup 接收区。
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, 1, num_max_tokens_per_rank);  // 跨 scaleout 发送 staging 区。
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleout_ranks,  // 跨 scaleout 接收区按来源 scaleout rank 切分。
                /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);  // 最坏 channel padding 容量。
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();  // hybrid dispatch 三段总字节数。
        }
    }

    static int64_t get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,  // token 上界、hidden、top-k。
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,  // 逻辑拓扑维度。
                                           const bool& is_scaleup_nvlink,  // scaleup 域是否走 NVLink。
                                           const bool& allow_multiple_reduction) {  // 是否允许多阶段 reduction。
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;  // 逻辑总 rank 数。
        const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);  // 单 token 在 combine buffer 中的结构。

        if (num_scaleout_ranks == 1) {  // direct combine 模式。
            // Direct combine 的 send_buffer 是把专家输出送回源 rank 的 staging；
            // allow_multiple_reduction=false 时每个 topk lane 都可能单独占一个 token slot，
            // 因此容量按 num_topk 放大。
            const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;  // reduce buffer 第一维容量。
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : num_ranks,  // NVLink 直写时不需要 send staging，否则按 rank 切分。
                // For single reduction cases, the maximum number of received tokens is
                // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
                num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));  // 单 reduction 模式按 top-k 放大容量。
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_tokens_in_layout, num_max_tokens_per_rank);  // combine 接收/归约区。
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();  // direct combine 总字节数。
        } else {
            // Hybrid combine 同样有 scale-up recv、scaleout recv、scaleout send 三段。
            // rank-layout/topk-layout 的第一维由 allow_multiple_reduction 决定，和
            // combine_utils.cuh::use_rank_layout 保持一致。
            const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;  // scaleup reduce layout 宽度。
            const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;  // scaleout reduce layout 宽度。
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);  // scaleup 阶段接收/归约区。
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);  // scaleout 阶段接收/归约区。
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, allow_multiple_reduction ? 1 : num_topk,  // 多阶段 reduction 可压缩发送布局第一维。
                /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
                num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));  // 跨 scaleout send staging 容量。
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();  // hybrid combine 三段总字节数。
        }
    }

    static int64_t calculate_buffer_size(const int64_t& nccl_comm,  // Python 传入的 NCCL communicator 句柄。
                                         const int& num_max_tokens_per_rank, const int& hidden,  // 每 rank token 上界和 hidden。
                                         int num_topk, const bool& use_fp8_dispatch,  // top-k 数和是否按 FP8 dispatch 估算。
                                         const bool& allow_hybrid_mode,  // 是否允许 hybrid 拓扑。
                                         const bool& allow_multiple_reduction) {  // combine 是否允许多阶段归约。
        // Python ElasticBuffer(num_bytes=None) 会调用这里估算 buffer payload 区大小。
        // 输入维度含义：
        //   num_max_tokens_per_rank：每个 rank 最多输入 token 数；
        //   hidden：每个 token 的 hidden 维度；
        //   num_topk：每个 token 选择几个 expert，0 表示未知时按 32 保守估计。
        // 返回值取 dispatch 与 combine 两种 layout 的最大值，避免同一 buffer 在不同阶段越界。
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);  // size hint 必须有正的 token 上界和 hidden。

        // The worst case SF bytes must be less than the main part
        EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);  // scale-factor 最坏字节数不能超过主 payload 粗略上界。

        // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
        num_topk = num_topk == 0 ? 32 : num_topk;  // num_topk 未知时按 32 保守估算。

        // Topology
        const auto [num_rdma_ranks, num_nvl_ranks] = nccl::get_physical_domain_size(nccl_comm);  // 查询物理 RDMA/NVLink 拓扑。
        const auto [num_scaleout_ranks, num_scaleup_ranks] = nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);  // 查询逻辑 scaleout/scaleup 拓扑。
        const auto is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;  // 逻辑 scaleup 是否完全落在 NVLink 域。

        // Dispatch size
        const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);  // dispatch payload 元素字节数。
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0; // An approximation for number of SF packs；FP8 scale pack 估算。
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,  // dispatch token/layout 参数。
            num_scaleout_ranks, num_scaleup_ranks,  // 逻辑拓扑参数。
            is_scaleup_nvlink);  // scaleup 是否 NVLink。

        // Combine layout
        const auto num_combine_bytes = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,  // combine token/layout 参数。
            num_scaleout_ranks, num_scaleup_ranks,  // 逻辑拓扑参数。
            is_scaleup_nvlink, allow_multiple_reduction);  // scaleup 链路和 reduction 模式。

        // Return the maximum of those layouts
        return std::max(num_dispatch_bytes, num_combine_bytes);  // payload buffer 需同时容纳 dispatch 和 combine 的最大需求。
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,          // recv_x 与可选 recv_x_scales。
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,  // recv_topk_idx / recv_topk_weights。
               std::optional<torch::Tensor>,                         // cloned_topk_idx，handle 需要防止用户修改原 topk。
               std::vector<int>,                                      // CPU per-local-expert 接收计数。
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,  // prefix/src/dst slot 等核心 handle tensor。
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,  // hybrid forward metadata 和 channel linked list。
               std::optional<EventHandle>>                            // async 模式通信事件。
    dispatch(const torch::Tensor& x,  // 输入 payload Tensor [num_tokens, hidden]。
             const std::optional<torch::Tensor>& sf,  // FP8 scale factor，可为空。
             const torch::Tensor& topk_idx,  // top-k expert 路由矩阵 [num_tokens,num_topk]。
             const std::optional<torch::Tensor>& topk_weights,  // top-k 权重，可为空。
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,  // 可选本地 expert 累计接收统计。
             const std::optional<int>& cached_num_recv_tokens,  // cached handle 中的接收 token 总数。
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,  // cached per-expert 接收计数。
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,  // cached scaleup rank 前缀和。
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,  // cached expert 前缀和。
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,  // cached 目标 slot 索引。
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,  // cached hybrid forward metadata。
             const std::optional<torch::Tensor>& cached_channel_linked_list,  // cached hybrid channel linked list。
             const int& num_max_tokens_per_rank,  // 每 rank token 上界。
             const int& num_experts, const int& expert_alignment,  // 全局 expert 数和 expert token 对齐。
             const int& num_sms, const int& num_qps,  // dispatch kernel 使用的 SM/QP 数。
             const std::optional<EventHandle>& previous_event,  // dispatch 前等待事件。
             const std::optional<EventHandle>& previous_event_before_epilogue,  // copy epilogue 前等待事件。
             const bool& async_with_compute_stream,  // 是否异步返回。
             const bool& allocate_on_comm_stream,  // 是否在 comm_stream 上分配输出。
             const bool& do_handle_copy, const bool& do_cpu_sync, const bool& do_expand,  // handle copy、CPU sync、expand layout 开关。
             const bool& use_tma_aligned_col_major_sf) const {  // FP8 scale factor 是否按 TMA 友好布局输出。
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);  // dispatch 至少需要一个 SM。

        // Cached mode must have responding handles
        // cached_* 来自 Python EPHandle。cached 模式跳过 layout 重新计算，只重发新的 x/sf payload。
        // 因此所有会影响地址计算的元数据必须同时存在；hybrid 模式还需要转发链表和 forward metadata。
        const bool cached_mode = cached_num_recv_tokens.has_value();  // 是否复用 Python EPHandle。
        if (cached_mode) {  // cached 模式需要所有 layout 元数据齐备。
            EP_HOST_ASSERT(cached_num_recv_tokens.has_value());  // 接收 token 数必须存在。
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());  // per-expert 接收计数必须存在。
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_scaleup_rank.has_value());  // scaleup 前缀和必须存在。
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());  // expert 前缀和必须存在。
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());  // 目标 slot 索引必须存在。

            // Hybrid kernels require more
            if (nccl_context->num_scaleout_ranks > 1) {  // hybrid 模式还需要跨 scaleout 转发 metadata。
                EP_HOST_ASSERT(cached_token_metadata_at_forward.has_value());  // forward metadata 必须存在。
                EP_HOST_ASSERT(cached_channel_linked_list.has_value());  // channel linked list 必须存在。
            }
        }

        // Check data tensor
        // x 的主维度为 [num_tokens, hidden]。hidden * element_size 必须按 int4 对齐，
        // 因为 device kernel 会用向量化 load/store 或 TMA 友好的布局搬运 token payload。
        const auto [num_tokens, hidden] = get_shape<2>(x);  // x 形状 [num_tokens,hidden]。
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());  // 单 token payload 字节数。
        const auto num_local_experts = num_experts / nccl_context->num_ranks;  // 本 rank 拥有的 expert 数。
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());  // x 必须是 CUDA contiguous tensor。
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);  // 每 token 字节数需 int4 对齐。
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);  // 输入 token 数不能超过 buffer 按照的上界。

        // Check SF stuffs
        // FP8 dispatch 时 sf 保存 scale factor，形状通常为 [num_tokens, hidden / 128] 或
        // TMA 对齐后的 column-major 变体。这里不假设 contiguous，而是把 token/hidden stride
        // 传给 kernel，让 device 端按真实 stride 读取。
        int num_sf_packs = 0;  // 每 token scale-factor pack 数，BF16 路径为 0。
        void* sf_ptr = nullptr;  // scale-factor 数据指针，BF16 路径为 nullptr。
        int sf_token_stride = 0, sf_hidden_stride = 0;  // scale-factor tensor 的两个 stride。
        if (sf.has_value()) {  // FP8 dispatch 路径。
            // SF must be FP32 or packed UE8M0x4
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());  // scale-factor 形状 [num_tokens,num_sf_packs]。
            EP_HOST_ASSERT(num_tokens == num_tokens_);  // scale-factor token 维必须和 x 一致。
            EP_HOST_ASSERT(sf->is_cuda());  // scale-factor 必须在 GPU 上。
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));  // scale pack 元素大小必须匹配 kernel 期望。
            num_sf_packs = num_sf_packs_;  // 保存每 token scale pack 数。
            sf_ptr = sf->data_ptr();  // 保存 scale-factor 裸指针。
            sf_token_stride = sf->stride(0);  // 保存 token 维 stride。
            sf_hidden_stride = sf->stride(1);  // 保存 hidden/pack 维 stride。
        }

        // Check top-k stuffs
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);  // topk_idx 形状 [num_tokens,num_topk]。
        EP_HOST_ASSERT(num_tokens == num_tokens_);  // topk_idx token 维必须和 x 一致。
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);  // dtype 必须是 C++ topk_idx_t 对应类型。
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());  // topk_idx 必须是 CUDA contiguous。

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;  // top-k 权重裸指针，可为空。
        if (topk_weights.has_value()) {  // 如果用户传了 dispatch 权重。
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());  // 权重形状 [num_tokens,num_topk]。
            EP_HOST_ASSERT(num_tokens == num_tokens__);  // 权重 token 维必须匹配。
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());  // 权重必须 CUDA contiguous。
            topk_weights_ptr = topk_weights->data_ptr<float>();  // 保存 float 权重指针。
        }

        // Expert receiving counter
        int* cumulative_local_expert_recv_stats_ptr = nullptr;  // 可选累计统计指针。
        if (cumulative_local_expert_recv_stats.has_value()) {  // 用户需要在线统计本地 expert 接收量。
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());  // 统计 tensor 形状 [num_local_experts]。
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());  // 统计 tensor 必须 CUDA contiguous。
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);  // 统计长度必须等于本地 expert 数。
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();  // 保存统计指针。
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);  // 建立 stream 依赖并可能切到 comm_stream 分配。

        // The number of received tokens per expert
        // psum_num_recv_tokens_per_expert 记录本 rank 每个本地 expert 的接收前缀和。
        // non-expand 模式用 inclusive prefix；expand 模式额外分配 num_local_experts + 1，
        // 方便表示“每个 expert 的起止 offset”，并支持 expert_alignment padding。
        // This is useful for expanding mode
        EP_HOST_ASSERT(num_experts % nccl_context->num_ranks == 0);  // expert 必须能均匀分配到 rank。
        auto psum_num_recv_tokens_per_expert = cached_psum_num_recv_tokens_per_expert.value_or(torch::Tensor());  // cached 或稍后新分配的 expert 前缀和。
        if (cached_mode) {  // cached 模式校验 handle 中的 expert 前缀和。
            const auto& [num_local_experts_] = get_shape<1>(psum_num_recv_tokens_per_expert);  // 前缀和长度。
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);  // 长度必须等于本地 expert 数。
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_cuda() and psum_num_recv_tokens_per_expert.is_contiguous());  // 必须 CUDA contiguous。
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);  // dtype 必须 int32。
        } else {
            // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
            psum_num_recv_tokens_per_expert = torch::empty(
                {num_local_experts + 1}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));  // 多分配 1 个元素，后面按模式 slice。
        }

        // The prefix sum tensor of number of received tokens from each rank
        // psum_num_recv_tokens_per_scaleup_rank: [num_scaleup_ranks]。
        // 它按 scaleup rank 聚合去重后的 token 数；combine 会用它判断 reduce_buffer 中
        // 哪些 token 属于哪个 scaleup peer，最后一个元素也常用于得到总接收 token 数。
        // Will also be used in combine as the dispatch handle
        auto psum_num_recv_tokens_per_scaleup_rank = cached_psum_num_recv_tokens_per_scaleup_rank.value_or(torch::Tensor());  // cached 或新分配 scaleup 前缀和。
        if (cached_mode) {  // cached 模式校验 scaleup 前缀和。
            const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);  // 前缀和长度。
            EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);  // 长度必须等于 scaleup rank 数。
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());  // 必须 CUDA contiguous。
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);  // dtype 必须 int32。
        } else {
            psum_num_recv_tokens_per_scaleup_rank = torch::empty(
                {nccl_context->num_scaleup_ranks}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));  // 新分配 [num_scaleup_ranks] 前缀和。
        }

        // Decide number of channels by shared memory consumption
        // Only for hybrid version
        int num_channels_per_sm = 1, num_channels = 1;  // direct 模式默认只有 1 个 channel。
        const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();  // 当前 GPU 每 block 可用动态 shared memory。
        if (nccl_context->num_scaleout_ranks > 1) {  // hybrid 模式需要按 SM/shared memory 计算 channel 数。
            const auto dispatch_token_layout = get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk);  // dispatch 单 token layout。
            const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);  // combine 单 token layout。
            EP_HOST_ASSERT(num_sms <= kNumMaxSMs);  // SM 数不能超过静态 channel 上界假设。
            num_channels_per_sm = std::min<int>(
                (num_smem_bytes - get_num_notify_smem_bytes(nccl_context->num_ranks, num_experts)) / dispatch_token_layout.get_num_bytes<true>(),  // dispatch token staging 可容纳 channel 数。
                32 - kNumNotifyWarps);  // 一个 block 最多 warp 数减去 notify warps。
            num_channels_per_sm = std::min<int>(
                num_smem_bytes / combine_token_layout.get_num_bytes<true>(),  // combine token staging 可容纳 channel 数。
                num_channels_per_sm);  // 同时满足 dispatch/combine 两种 layout。
            num_channels_per_sm = std::min<int>(
                /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);  // dispatch/forward 两类 warp 共享 channel 槽。
            if (not prefer_overlap_with_compute)  // 不追求 overlap 时限制每 SM channel 数，避免过度切分。
                num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);  // 每 SM 最多 4 channel。
            num_channels = num_sms * num_channels_per_sm;  // 总 channel 数。
            if (get_env<int>("EP_BUFFER_DEBUG"))  // debug 模式打印 channel 选择。
                printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);  // 输出每 SM channel 数。
        }

        // Non-hybrid mode handles
        std::optional<torch::Tensor> deterministic_rank_count_buffer = std::nullopt;  // deterministic prologue 临时 rank count buffer。
        auto dst_buffer_slot_idx = cached_dst_buffer_slot_idx.value_or(torch::Tensor());  // 目标 buffer slot 索引，cached 或新建。
        if (nccl_context->num_scaleout_ranks == 1) {  // direct/non-hybrid 模式。
            if (cached_mode) {  // cached 模式校验 dst slot 矩阵。
                const auto [num_tokens__, num_topk_] = get_shape<2>(dst_buffer_slot_idx);  // 形状 [num_tokens,num_topk]。
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk_);  // 形状必须匹配当前输入。
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());  // 必须 CUDA contiguous。
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);  // dtype 必须 int32。
            } else if (deterministic) {  // deterministic 模式先运行 prologue 计算稳定 slot。
                const auto prologue_num_sms = jit::device_runtime->get_num_sms();  // prologue 使用全卡 SM。

                // Allocate new tensors
                deterministic_rank_count_buffer = torch::empty(
                    {prologue_num_sms, nccl_context->num_scaleup_ranks},  // 每个 SM/scaleup rank 一个计数槽。
                    torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));  // int32 计数 buffer。
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));  // 每个 token/topk 的目标 slot。

                // Launch a kernel to preprocess the destination slot indices
                launch_dispatch_deterministic_prologue(topk_idx.data_ptr<topk_idx_t>(),  // 输入路由矩阵。
                                                       deterministic_rank_count_buffer->data_ptr<int>(),  // rank count 临时 buffer。
                                                       dst_buffer_slot_idx.data_ptr<int>(),  // 输出目标 slot 索引。
                                                       num_tokens, num_max_tokens_per_rank,  // token 数和上界。
                                                       num_experts, num_topk,  // expert 总数和 top-k。
                                                       nccl_context->scaleup_rank_idx, nccl_context->num_scaleup_ranks,  // 本 rank scaleup 坐标。
                                                       prologue_num_sms,  // prologue 使用 SM 数。
                                                       jit::device_runtime->get_num_smem_bytes(),  // shared memory 上限。
                                                       comm_stream);  // 在通信 stream 上运行。
            } else {
                // Allocate a new tensor
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));  // 非 deterministic 由主 dispatch kernel 填 slot。
            }
        }

        // Hybrid mode handles
        std::optional<torch::Tensor> token_metadata_at_forward, channel_linked_list;  // hybrid 模式需要返回给 handle 的两类 metadata。
        int *token_metadata_at_forward_ptr = nullptr, *channel_linked_list_ptr = nullptr;  // 传给 kernel 的 metadata 裸指针。
        if (nccl_context->num_scaleout_ranks > 1) {  // hybrid 模式。
            EP_HOST_ASSERT(not deterministic);  // 当前 hybrid path 不支持 deterministic prologue。

            // The token destination slot idx during forward
            // `[i, j, k, l]` means: from channel i from scale-out peer k, the j-th token's index in the l-th rank buffer
            // 中文：这个 4D 表主要让 cached dispatch 复用 forward 阶段的目标 slot 分配。
            // NOTES: Used primarily for cached mode
            // TODO: May make it a linked list to remove the redundant info in `token_metadata_at_forward`
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);  // 每 channel 最坏 token 容量。
            if (cached_mode) {  // cached 模式校验 dst_buffer_slot_idx 的 4D hybrid layout。
                const auto [num_channels_, num_scaleout_ranks_, num_max_tokens_per_channel_, num_topk_] =
                    get_shape<4>(dst_buffer_slot_idx);  // 形状 [channels,scaleout,max_tokens_per_channel,topk]。
                EP_HOST_ASSERT(num_channels == num_channels_ and nccl_context->num_scaleout_ranks == num_scaleout_ranks_ and
                               num_max_tokens_per_channel == num_max_tokens_per_channel_ and num_topk == num_topk_);  // cached shape 必须完全匹配当前配置。
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());  // 必须 CUDA contiguous。
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);  // dtype 必须 int32。
            } else {
                // 中文：非 cached 模式分配新的 dst slot 4D tensor，后续 dispatch kernel 写入。
                dst_buffer_slot_idx = torch::empty(
                    {num_channels, nccl_context->num_scaleout_ranks, num_max_tokens_per_channel, num_topk},  // hybrid dst slot 4D layout。
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)  // int32 CUDA tensor。
                );  // 主 dispatch kernel 会填充它。
            }

            // The token metadata during forward
            // `[i, j]` means: in channel i, the j-th forwarded token's metadata
            // Info contains:
            // 中文：该 3D tensor 是 hybrid combine 反向追踪 forward token 的关键链路元数据。
            //   - Scaleout rank index and source token index in the original rank (0)
            // 中文：第 0 组字段告诉 combine 这个 forwarded token 最终应该回到哪个 scaleout/source token。
            //   - Whether the token is the last one in the chunk (1)
            //   - cached top-k scaleup peer indices (top-k)
            //   - each selections' destination slot indices (top-k)
            const auto num_max_forwarded_tokens = nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1;  // 每 channel forward metadata 容量，+1 作为哨兵/尾节点。
            const auto num_forward_metadata_dims = 2 + num_topk * 2;  // 每个 forward token metadata 的字段数。
            if (cached_mode) {  // cached 模式直接复用 handle 中的 forward metadata。
                token_metadata_at_forward = cached_token_metadata_at_forward;  // 保存 optional tensor。
                const auto [num_channels_, num_max_forwarded_tokens_, num_forward_metadata_dims_] = get_shape<3>(token_metadata_at_forward.value());  // 校验 3D shape。
                EP_HOST_ASSERT(num_channels == num_channels_ and num_max_forwarded_tokens == num_max_forwarded_tokens_
                               and num_forward_metadata_dims == num_forward_metadata_dims_);  // shape 必须匹配当前 top-k/channel 配置。
                EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());  // 必须 CUDA contiguous。
                EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);  // dtype 必须 int32。
            } else {
                // 中文：非 cached 模式分配新的 forward metadata tensor，供 hybrid combine 反向使用。
                token_metadata_at_forward = torch::empty(
                    {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims},  // [channel,forwarded_token,metadata_dim]。
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)  // int32 CUDA tensor。
                );  // 主 dispatch kernel 填充 forward metadata。
            }
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();  // 保存裸指针供 kernel 写入/读取。

            // Per-scaleup-peer-per-channel linked list
            // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
            if (cached_mode) {  // cached 模式复用 channel linked list。
                channel_linked_list = cached_channel_linked_list;  // 保存 optional tensor。
                const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());  // 校验 3D shape。
                channel_linked_list_ptr = channel_linked_list->data_ptr<int>();  // 先取指针，后续 assert 失败也会中断。
                EP_HOST_ASSERT(num_channels == num_channels__);  // channel 数必须一致。
                EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);  // token 容量维必须一致。
                EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);  // scaleup peer 维必须一致。
                EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());  // 必须 CUDA contiguous。
                EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);  // dtype 必须 int32。
            } else {
                channel_linked_list = torch::empty(
                    // Index 0 of the list means the starting item
                    {num_channels,  // channel 维。
                    nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1,  // 每 channel token 链表容量，0 为头节点。
                    nccl_context->num_scaleup_ranks},  // scaleup peer 维。
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)  // int32 CUDA tensor。
                );  // 主 dispatch/copy epilogue 会维护链表。
            }
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();  // 保存裸指针供 kernel 使用。
        }

        // Clone `topk_idx` for saving in the handle (to prevent users' modification)
        auto copied_topk_idx = std::optional<torch::Tensor>();  // 返回给 Python handle 的 cloned topk_idx。
        topk_idx_t* copied_topk_idx_ptr = nullptr;  // clone 输出指针；不 clone 时为 nullptr。
        if (do_handle_copy and not cached_mode) {  // 非 cached 模式且用户要求保护 handle。
            copied_topk_idx = torch::empty_like(topk_idx);  // 分配和 topk_idx 同 shape/dtype 的 tensor。
            copied_topk_idx_ptr = copied_topk_idx->data_ptr<topk_idx_t>();  // dispatch kernel 会把 topk_idx 拷入这里。
        }

        // Check buffer size
        EP_HOST_ASSERT(get_dispatch_buffer_size(
                       num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, x.element_size(),  // dispatch layout shape。
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,  // 逻辑拓扑。
                       nccl_context->is_scaleup_nvlink) <= num_buffer_bytes);  // payload buffer 必须足够。

        // Ready and clean host workspace for this round
        const auto host_workspace_layout = layout::WorkspaceLayout(
            host_workspace,  // CPU pinned workspace 基址。
            nccl_context->num_scaleout_ranks,  // scaleout 维度。
            nccl_context->num_scaleup_ranks,  // scaleup 维度。
            num_experts);  // expert 总数，决定 expert count 区域偏移。
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), nccl_context->num_scaleup_ranks, 0);  // 清空 rank count CPU 镜像。
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);  // 清空 expert count CPU 镜像。
        std::atomic_thread_fence(std::memory_order_seq_cst);  // 确保 CPU 清零对后续 mapped GPU 写/CPU 轮询顺序可见。

        // Do dispatch into the buffers (with SM limitation)
        EP_HOST_ASSERT(num_sms <= jit::device_runtime->get_num_sms());  // 使用 SM 数不能超过设备 SM 总数。
        launch_dispatch(x.data_ptr(), sf_ptr,  // 输入 payload 和可选 scale-factor。
                        topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,  // 路由矩阵和可选权重。
                        copied_topk_idx_ptr,  // 可选 cloned topk_idx 输出。
                        cumulative_local_expert_recv_stats_ptr,  // 可选累计统计输出。
                        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),  // scaleup rank 前缀和输出。
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),  // expert 前缀和输出。
                        dst_buffer_slot_idx.data_ptr<int>(),  // 目标 slot 输出/输入。
                        token_metadata_at_forward_ptr,  // hybrid forward metadata 输出/输入。
                        num_tokens, num_max_tokens_per_rank,  // token 数和上界。
                        hidden, x.element_size(),  // hidden 和元素字节数。
                        num_sf_packs, sf_token_stride, sf_hidden_stride,  // scale-factor layout。
                        num_experts, num_topk, expert_alignment,  // expert/top-k/对齐配置。
                        nccl_context->dev_comm, nccl_context->window,  // NCCL device communicator 和 window。
                        buffer,  // payload buffer。
                        workspace, mapped_host_workspace,  // GPU workspace 和 device-mapped host workspace。
                        nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,  // 当前 rank 逻辑坐标。
                        nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,  // 逻辑拓扑维度。
                        nccl_context->is_scaleup_nvlink,  // scaleup 是否 NVLink。
                        num_sms, num_channels_per_sm,  // SM 数和每 SM channel 数。
                        num_smem_bytes,  // shared memory 上限。
                        num_qps, num_gpu_timeout_cycles,  // QP 数和 GPU 超时。
                        cached_mode, deterministic, do_cpu_sync,  // cached/deterministic/CPU sync 模式。
                        comm_stream);  // 在 comm_stream 上启动。

        // Received token counters
        int num_recv_tokens = 0, num_expanded_tokens = 0;  // 接收 token 总数与 expand 后 token slot 数。
        int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;  // CPU 轮询进度游标。
        std::vector<int> num_recv_tokens_per_expert_list;  // Python 返回的 per-expert 接收计数。

        // Assign these values according to modes
        if (cached_mode) {  // cached 模式直接复用 handle 中的计数。
            // Cached mode
            // TODO: support to expand for MoE training backward with cached handles from non-expanding forward,
            // which requires maintaining the same expanding order between forward and backward
            EP_HOST_ASSERT(not do_expand and "Cannot do expand with cached mode");  // cached 模式当前不支持重新 expand。
            EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");  // cached 模式不重新等待 CPU 计数。
            num_recv_tokens = cached_num_recv_tokens.value();  // 复用总接收 token 数。
            num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();  // 复用 per-expert 计数。
        } else if (do_cpu_sync) {
            // Non-cached mode with sync
            const auto start_cpu_time = std::chrono::high_resolution_clock::now();  // 记录 CPU 轮询起始时间，用于超时。
            while (true) {  // 轮询 mapped host workspace，等待 GPU 写回计数。
                bool ready = true;  // 当前轮是否已读到所有 ready 计数。

                // Read number of received tokens from each scaleup rank
                while (counter_scaleup_rank_idx < nccl_context->num_scaleup_ranks and ready) {  // 依次读取 rank count。
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_rank_count_ptr<false>()[counter_scaleup_rank_idx]);  // 解码 GPU 写入的 encoded count。
                    if ((ready = math::is_decoded_positive_ready(count))) {  // 非负表示该 count 已 ready。
                        num_recv_tokens += count;  // 累加总接收 token 数。
                        ++ counter_scaleup_rank_idx;  // 推进 rank count 游标。
                    }
                }

                // Read expert counts
                while (counter_local_expert_idx < num_local_experts and ready) {  // 依次读取本地 expert count。
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_expert_count_ptr<false>()[counter_local_expert_idx]);  // 解码 expert count。
                    if ((ready = math::is_decoded_positive_ready(count))) {  // 该 expert count 已 ready。
                        num_recv_tokens_per_expert_list.push_back(count);  // 记录真实 per-expert 接收 token 数。
                        num_expanded_tokens += count;  // expand 容量先累加真实 token。
                        ++ counter_local_expert_idx;  // 推进 expert count 游标。
                    }
                }

                // Ready and do next steps
                const auto get_buffer_info = [&]() {  // 构造 debug/timeout 时的 workspace 计数字符串。
                    std::stringstream ss;  // 输出 rank count 和 expert count。
                    ss << "CPU side received count (scaleup: " << nccl_context->scaleup_rank_idx << "): ";  // 打印当前 scaleup rank。
                    for (int i = 0; i < nccl_context->num_scaleup_ranks + num_local_experts; ++ i) {  // 遍历 rank count + expert count 连续区域。
                        ss << host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i];  // 输出原始 encoded 值。
                        ss << (i == nccl_context->num_scaleup_ranks - 1 ? " # ": " ");  // rank/expert 区域之间用 # 分隔。
                    }
                    return ss.str();  // 返回完整 debug 字符串。
                };  // lambda 结束。
                if (ready) {  // 所有计数都 ready。
                    if (get_env<int>("EP_BUFFER_DEBUG"))  // debug 模式打印计数。
                        printf("%s\n", get_buffer_info().c_str());  // 输出计数信息。
                    break;  // 退出 CPU 轮询。
                }

                // Timeout checks
                const auto now = std::chrono::high_resolution_clock::now();  // 当前时间。
                if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > num_cpu_timeout_secs)  // 超过 CPU 等待阈值。
                    throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());  // 抛出带 workspace 状态的异常。
            }
        } else {
            // Non-cached mode without CPU sync, allocate with the worst case
            num_recv_tokens = num_max_tokens_per_rank * nccl_context->num_ranks;  // 最坏每个 rank 都向本 rank 发满 token。
            num_expanded_tokens = nccl_context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_local_experts);  // expand 最坏 top-k 有效槽数。
            num_expanded_tokens += (expert_alignment - 1) * num_local_experts;  // 为每个 expert 预留 alignment padding。
            num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);  // 总 expand slot 数按 expert_alignment 对齐。
        }

        // Allocate received tensors
        // `recv_src_metadata` includes source token indices and buffer slot indices
        const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;  // 输出行数取决于 expand 模式。
        auto recv_x = torch::empty({num_allocated_tokens, hidden}, x.options());  // 分配接收 payload [allocated_tokens,hidden]。
        auto recv_sf = std::optional<torch::Tensor>();  // 可选 FP8 scale-factor 输出。
        auto recv_topk_idx = std::optional<torch::Tensor>();  // 可选接收 top-k 路由输出。
        auto recv_topk_weights = std::optional<torch::Tensor>();  // 可选接收权重输出。
        auto recv_src_metadata = torch::empty(
            {num_recv_tokens, num_topk + 2},  // 源 metadata 形状 [num_recv_tokens,num_topk+2]。
            torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));  // int32 CUDA tensor。

        // Optional tensors
        void* recv_sf_ptr = nullptr;  // FP8 scale-factor 输出指针。
        topk_idx_t* recv_topk_idx_ptr = nullptr;  // 接收 top-k 路由输出指针。
        float* recv_topk_weights_ptr = nullptr;  // 接收权重输出指针。
        int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;  // 接收 scale-factor 输出 stride。
        if (sf.has_value()) {  // FP8 dispatch 需要分配接收 scale-factor。
            if (not use_tma_aligned_col_major_sf) {  // 普通行主布局。
                recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;  // [token,pack] 连续布局。
            } else {
                // TMA-aligned layout for the next GEMM input
                recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks);  // 列主 + token 维对齐布局。
            }
            recv_sf = torch::empty_strided({num_allocated_tokens, num_sf_packs},
                                           {recv_sf_token_stride, recv_sf_hidden_stride},  // 使用真实 stride 构造 scale-factor 输出。
                                           sf->options());  // dtype/device 与输入 sf 一致。
            recv_sf_ptr = recv_sf->data_ptr();  // 保存输出 scale-factor 指针。
        }
        if (not do_expand) {  // 非 expand 模式需要保留每行接收 token 的 top-k 路由矩阵。
            recv_topk_idx = torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());  // 分配 [allocated_tokens,num_topk]。
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();  // 保存输出 topk_idx 指针。
        }
        if (topk_weights.has_value()) {  // 用户传了 top-k 权重时输出对应权重。
            recv_topk_weights = do_expand ?
                torch::empty({num_allocated_tokens}, topk_weights->options()) :  // expand 模式每行只对应一个 expert 槽，权重是一维。
                torch::empty({num_allocated_tokens, num_topk}, topk_weights->options());  // 非 expand 模式保留 [token,topk]。
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();  // 保存输出权重指针。
        }

        // Process prefix sum, in expanding mode, it is also atomic counters
        if (do_expand) {
            // Slice and exclusive part and do atomic additions into inclusive
            EP_HOST_ASSERT(not cached_mode);  // cached 模式不支持 expand。
            psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);  // expand 使用 exclusive prefix 部分。
        } else if (not cached_mode) {
            // Slice the inclusive part (and will not be used in the epilogue)
            psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);  // 非 expand 返回 inclusive prefix 部分。
        }
        EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) == num_local_experts);  // 返回给 Python 的 expert 前缀和长度必须为本地 expert 数。

        // Launch copy kernels with full SMs
        stream_control_before_epilogue(previous_event_before_epilogue);  // 如果用户要求 epilogue 等待额外事件，在这里插入依赖。
        launch_dispatch_copy_epilogue(buffer, workspace,  // 从通信 buffer/workspace 拷贝到输出 tensor。
                                      psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),  // scaleup rank 前缀和。
                                      psum_num_recv_tokens_per_expert.data_ptr<int>(),  // expert 前缀和。
                                      recv_x.data_ptr(), recv_sf_ptr,  // 输出 payload 和可选 scale-factor。
                                      recv_topk_idx_ptr, recv_topk_weights_ptr,  // 输出路由和权重。
                                      recv_src_metadata.data_ptr<int>(),  // 输出 combine 所需源 metadata。
                                      channel_linked_list_ptr,  // hybrid 模式链表 metadata。
                                      num_recv_tokens, num_max_tokens_per_rank,  // 接收 token 数和每 rank 上界。
                                      num_hidden_bytes,  // 单 token payload 字节数。
                                      num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,  // scale-factor 输出布局。
                                      num_experts, num_topk,  // expert 总数和 top-k。
                                      nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,  // 当前 rank 逻辑坐标。
                                      nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,  // 逻辑拓扑。
                                      jit::device_runtime->get_num_sms(),  // copy epilogue 使用全卡 SM。
                                      jit::device_runtime->get_num_smem_bytes(),  // shared memory 上限。
                                      num_channels,  // hybrid channel 总数。
                                      do_expand, cached_mode,  // expand 和 cached 模式。
                                      comm_stream);  // epilogue 在通信 stream 上运行。

        // Stream control
        const auto event = stream_control_epilogue(
            {x, sf, topk_idx, topk_weights,  // 输入 tensor 保活/record_stream。
             recv_x, recv_sf, recv_topk_idx, recv_topk_weights,  // 输出 tensor 保活/record_stream。
             cumulative_local_expert_recv_stats,  // 可选统计 tensor。
             copied_topk_idx,  // 可选 cloned topk_idx。
             psum_num_recv_tokens_per_scaleup_rank,  // scaleup 前缀和。
             psum_num_recv_tokens_per_expert,  // expert 前缀和。
             recv_src_metadata,  // combine handle metadata。
             deterministic_rank_count_buffer,  // deterministic prologue 临时 buffer。
             dst_buffer_slot_idx,  // dispatch slot metadata。
             token_metadata_at_forward,  // hybrid forward metadata。
             channel_linked_list},  // hybrid channel linked list。
            compute_stream,  // 原 compute stream。
            allocate_on_comm_stream, async_with_compute_stream);  // 分配归属和异步策略。

        return {recv_x, recv_sf,  // 接收 payload 和可选 scale-factor。
                recv_topk_idx, recv_topk_weights,  // 接收路由和权重。
                copied_topk_idx,  // handle 中保存的 cloned topk_idx。
                num_recv_tokens_per_expert_list,  // CPU per-expert 接收计数。
                psum_num_recv_tokens_per_scaleup_rank,  // scaleup 前缀和。
                psum_num_recv_tokens_per_expert,  // expert 前缀和。
                recv_src_metadata,  // combine 源 metadata。
                dst_buffer_slot_idx,  // cached dispatch slot metadata。
                token_metadata_at_forward,  // cached hybrid forward metadata。
                channel_linked_list,  // cached hybrid linked list。
                event};  // 可选异步事件。
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    combine(const torch::Tensor& x,  // expert 计算后的本地 token Tensor [num_tokens,hidden]。
            const std::optional<torch::Tensor>& topk_weights,  // 可选 top-k 权重 [num_tokens,num_topk]。
            const std::optional<torch::Tensor>& bias_0,  // 可选 bias 0。
            const std::optional<torch::Tensor>& bias_1,  // 可选 bias 1。
            const torch::Tensor& src_metadata,  // dispatch 返回的源 metadata [num_reduced_tokens,num_topk+2]。
            const torch::Tensor& combined_topk_idx,  // 原始 topk_idx [num_combined_tokens,num_topk]。
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,  // dispatch scaleup 前缀和 handle。
            const std::optional<torch::Tensor>& token_metadata_at_forward,  // hybrid forward metadata。
            const std::optional<torch::Tensor>& channel_linked_list,  // hybrid linked list。
            const int& num_experts,  // 全局 expert 总数。
            const int& num_max_tokens_per_rank,  // 每 rank token 上界。
            const int& num_sms, const int& num_qps,  // combine 使用 SM/QP 数。
            const std::optional<EventHandle>& previous_event,  // combine 前等待事件。
            const std::optional<EventHandle>& previous_event_before_epilogue,  // reduce epilogue 前等待事件。
            const bool& async_with_compute_stream,  // 是否异步返回。
            const bool& allocate_on_comm_stream,  // 是否在 comm_stream 上分配输出。
            const bool& use_expanded_layout) const {  // dispatch 是否使用 expand layout。
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);  // combine 至少需要一个 SM。

        // Check data
        const auto [num_tokens, hidden] = get_shape<2>(x);  // expert 输出形状 [num_tokens,hidden]。
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());  // x 必须 CUDA contiguous。
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);  // combine 当前只支持 BF16 expert 输出。
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);  // 每 token 字节数需 int4 对齐。

        // Check tensors at dispatch
        const auto [num_combined_tokens, num_topk] = get_shape<2>(combined_topk_idx);  // 原始 token/topk 形状。
        const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);  // scaleup 前缀和长度。
        EP_HOST_ASSERT(combined_topk_idx.is_cuda() and combined_topk_idx.is_contiguous());  // topk_idx 必须 CUDA contiguous。
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);  // topk_idx dtype 必须匹配。
        EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);  // 前缀和长度必须等于 scaleup rank 数。
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());  // 前缀和必须 CUDA contiguous。
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);  // 前缀和 dtype 必须 int32。
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);  // 原始 token 数不能超过 dispatch 上界。

        // Check metadata
        // src_metadata 来自 dispatch 返回的 recv_src_metadata，形状为 [num_reduced_tokens, num_topk + 2]：
        //   第 0 列通常保存原 token index / slot；
        //   第 1 列保存 rank/top-k 相关槽位；
        //   后续 num_topk 列保存原 token 的 top-k expert 信息。
        // combine device kernel 依赖它把专家输出写回原 rank 的正确 token/top-k 位置。
        // For reduction mode, `num_tokens_` means the number of unexpanded tokens
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);  // src_metadata shape。
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));  // 非 expand 时 metadata 行数应等于输入 token 数。
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);  // metadata 列数必须是 top-k 加两个固定字段。
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());  // metadata 必须 CUDA contiguous。
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);  // metadata dtype 必须 int32。

        // Check optional tensors
        if (use_expanded_layout) {  // expand 模式每行已对应单个 expert 槽。
            // Reduction should be done with SwiGLU
            EP_HOST_ASSERT(not topk_weights.has_value());  // expand 模式不支持再用 [token,topk] 权重归约。
        } else if (topk_weights.has_value()) {  // 非 expand 模式可按 top-k 权重加权。
            const auto [num_tokens__, num_topk__] = get_shape<2>(topk_weights.value());  // 权重形状。
            EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);  // 权重 shape 必须匹配输入 token/topk。
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());  // 权重必须 CUDA contiguous。
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);  // 权重 dtype 必须 float32。
        }

        const auto bias_opts = std::vector({bias_0, bias_1});  // 把两个 optional bias 放入 vector 方便循环校验。
        void* bias_ptrs[2] = {nullptr, nullptr};  // 传给 kernel 的两个 bias 指针，默认空。
        for (int i = 0; i < 2; ++ i) {  // 最多两个 bias 输入。
            if (bias_opts[i].has_value()) {  // 当前 bias 存在。
                auto bias = bias_opts[i].value();  // 取出 bias tensor。
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());  // bias 必须二维且 contiguous。
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());  // bias dtype 必须和 x 一致。
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);  // bias shape 必须 [num_combined_tokens,hidden]。
                bias_ptrs[i] = bias.data_ptr();  // 保存 bias 裸指针。
            }
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);  // 建立 stream 依赖并可能切到 comm_stream 分配。

        // Check buffer size
        EP_HOST_ASSERT(get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk,
                                               nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,  // 逻辑拓扑。
                                               nccl_context->is_scaleup_nvlink, allow_multiple_reduction) <= num_buffer_bytes);  // payload buffer 必须足够。

        // Optional configs and metadata for hybrid combine
        // hybrid 模式下 dispatch 可能经历 scaleout(RDMA) + scaleup(NVLink) 两阶段。
        // token_metadata_at_forward 和 channel_linked_list 记录 forward 阶段跨 channel/peer 的链表，
        // combine 反向通信时必须沿用同一拓扑关系，否则会丢失某些 scaleup peer 的贡献。
        int num_channels = 1;  // direct 模式只有一个 channel。
        int* token_metadata_at_forward_ptr = nullptr;  // hybrid forward metadata 指针。
        int* channel_linked_list_ptr = nullptr;  // hybrid linked list 指针。
        if (nccl_context->num_scaleout_ranks > 1) {  // hybrid combine 需要 dispatch 保存的 metadata。
            // The token metadata during forward
            const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward.value());  // forward metadata shape。
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);  // 每 channel token 上界。
            num_channels = num_channels_;  // 从 metadata 中恢复 dispatch 时 channel 数。
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();  // 保存 forward metadata 指针。
            EP_HOST_ASSERT(d1 == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);  // token 容量维必须匹配。
            EP_HOST_ASSERT(d2 == 2 + num_topk * 2);  // metadata 字段数必须匹配 top-k。
            EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());  // 必须 CUDA contiguous。
            EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);  // dtype 必须 int32。

            // Per-scaleup-peer-per-channel linked list
            const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());  // linked list shape。
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();  // 保存 linked list 指针。
            EP_HOST_ASSERT(num_channels == num_channels__);  // channel 数必须匹配 forward metadata。
            EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);  // token 容量维必须匹配。
            EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);  // scaleup peer 维必须匹配。
            EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());  // 必须 CUDA contiguous。
            EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);  // dtype 必须 int32。
        }

        // Push data into remote buffers
        // launch_combine 先把本 rank expert output 写入目标 rank 可见的 reduce_buffer。
        // 这一步只完成“远端投递”，真正把多个 expert/top-k 贡献加权聚合到 combined_x 的工作
        // 在下面的 combine_reduce_epilogue 中完成。
        // NOTES: we don't use `num_hidden_bytes` due to enable later quantization possibility
        const auto reduce_buffer = launch_combine(
            x.data_ptr(),  // expert 输出 payload。
            topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,  // 可选 top-k 权重。
            src_metadata.data_ptr<int>(),  // dispatch 源 metadata。
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),  // scaleup rank 前缀和。
            token_metadata_at_forward_ptr,  // hybrid forward metadata。
            channel_linked_list_ptr,  // hybrid linked list。
            nccl_context->dev_comm, nccl_context->window,  // NCCL device communicator 和 window。
            buffer, workspace,  // payload buffer 和 workspace。
            num_reduced_tokens, num_max_tokens_per_rank,  // reduce 输入 token 数和上界。
            hidden, num_experts, num_topk,  // hidden、expert 总数、top-k。
            num_qps, num_gpu_timeout_cycles,  // QP 数和 GPU 超时。
            nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,  // 逻辑拓扑。
            nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,  // 当前 rank 逻辑坐标。
            nccl_context->is_scaleup_nvlink,  // scaleup 是否 NVLink。
            num_sms, jit::device_runtime->get_num_smem_bytes(),  // SM 数和 shared memory 上限。
            num_channels,  // hybrid channel 数。
            use_expanded_layout, allow_multiple_reduction,  // expand 和多阶段 reduction 开关。
            comm_stream);  // 在通信 stream 上运行。

        // Allocate output tensors
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());  // 分配最终输出 [num_combined_tokens,hidden]。
        auto combined_topk_weights = std::optional<torch::Tensor>();  // 可选归约后的 top-k 权重。
        float* combined_topk_weights_ptr = nullptr;  // 可选输出权重指针。
        if (topk_weights.has_value()) {  // 输入权重存在时输出 combined 权重。
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());  // 分配 [num_combined_tokens,num_topk]。
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();  // 保存输出权重指针。
        }

        // Combine pushed data
        stream_control_before_epilogue(previous_event_before_epilogue);  // 如果用户提供 epilogue 前事件，在 reduce 前等待。
        launch_combine_reduce_epilogue(combined_x.data_ptr(),  // 最终 combined_x 输出。
                                       combined_topk_weights_ptr,  // 可选 combined top-k weights 输出。
                                       combined_topk_idx.data_ptr<topk_idx_t>(),  // 原始 topk_idx。
                                       num_combined_tokens, num_max_tokens_per_rank,  // 原始 token 数和上界。
                                       hidden,  // hidden 维度。
                                       num_experts, num_topk,  // expert 总数和 top-k。
                                       reduce_buffer,  // launch_combine 返回的 reduce buffer 指针。
                                       bias_ptrs[0], bias_ptrs[1],  // 可选 bias 指针。
                                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,  // 逻辑拓扑。
                                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,  // 当前 rank 逻辑坐标。
                                       jit::device_runtime->get_num_sms(),  // epilogue 使用全卡 SM。
                                       jit::device_runtime->get_num_smem_bytes(),  // shared memory 上限。
                                       use_expanded_layout, allow_multiple_reduction,  // expand 和多阶段 reduction 开关。
                                       comm_stream);  // 在通信 stream 上运行 epilogue。

        // Stream control
        const auto event = stream_control_epilogue(
            {x, topk_weights, bias_0, bias_1,  // 输入 tensor 和可选 bias/weights。
             src_metadata,  // dispatch 源 metadata。
             combined_topk_idx,  // 原始 topk_idx。
             combined_x, combined_topk_weights,  // 输出 tensor。
             psum_num_recv_tokens_per_scaleup_rank,  // scaleup 前缀和 handle。
             token_metadata_at_forward,  // hybrid forward metadata。
             channel_linked_list},  // hybrid linked list。
            compute_stream,  // 原 compute stream。
            allocate_on_comm_stream, async_with_compute_stream);  // stream 分配和异步策略。
        return {combined_x, combined_topk_weights, event};  // 返回输出和可选事件。
    }
};

static void register_apis(pybind11::module_& m) {  // 注册 ElasticBuffer 相关 pybind API。
    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")  // 在 deep_ep._C 中暴露 ElasticBuffer 类。
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool, int, int, int, int, bool>())  // 注册构造函数签名。
        .def("destroy", &ElasticBuffer::destroy)  // 暴露 destroy。
        .def("get_comm_stream", &ElasticBuffer::get_comm_stream)  // 暴露通信 stream 查询。
        .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)  // 暴露物理拓扑查询。
        .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)  // 暴露逻辑拓扑查询。
        .def("barrier", &ElasticBuffer::barrier)  // 暴露 barrier。
        .def("engram_write", &ElasticBuffer::engram_write)  // 暴露 Engram 写入。
        .def("engram_fetch", &ElasticBuffer::engram_fetch)  // 暴露 Engram fetch。
        .def("pp_set_config", &ElasticBuffer::pp_set_config)  // 暴露 PP 配置。
        .def("pp_send", &ElasticBuffer::pp_send)  // 暴露 PP send。
        .def("pp_recv", &ElasticBuffer::pp_recv)  // 暴露 PP recv。
        .def("create_agrs_session", &ElasticBuffer::create_agrs_session)  // 暴露 AGRS session 创建。
        .def("destroy_agrs_session", &ElasticBuffer::destroy_agrs_session)  // 暴露 AGRS session 销毁。
        .def("agrs_set_config", &ElasticBuffer::agrs_set_config)  // 暴露 AGRS 配置。
        .def("agrs_get_inplace_tensor", &ElasticBuffer::agrs_get_inplace_tensor)  // 暴露 AGRS inplace tensor 申请。
        .def("all_gather", &ElasticBuffer::all_gather)  // 暴露 AGRS all_gather。
        .def("dispatch", &ElasticBuffer::dispatch)  // 暴露 EP dispatch。
        .def("combine", &ElasticBuffer::combine);  // 暴露 EP combine。
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);  // 暴露 buffer size hint 静态函数。

    // NCCL communicator handle
    m.def("get_local_nccl_unique_id", &nccl::get_local_unique_id);  // 暴露生成 NCCL unique id。
    m.def("create_nccl_comm", &nccl::create_nccl_comm);  // 暴露创建 NCCL communicator。
    m.def("destroy_nccl_comm", &nccl::destroy_nccl_comm);  // 暴露销毁 NCCL communicator。

    // Communication domain utilities
    m.def("get_physical_domain_size", &nccl::get_physical_domain_size);  // 暴露物理域大小查询。
    m.def("get_logical_domain_size", &nccl::get_logical_domain_size);  // 暴露逻辑域大小查询。
}

}  // namespace deep_ep  // 结束 Elastic/V2 C++ runtime 命名空间。
