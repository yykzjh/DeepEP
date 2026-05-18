#pragma once  // NCCL Gin handle wrapper 被多处 device kernel include，防止重复定义。

#include <nccl.h>         // NCCL host/device 基础类型。
#include <nccl_device.h>  // ncclGin、team tag、device communicator API。

#include <deep_ep/common/exception.cuh>  // EP_DEVICE_ASSERT 等设备端检查宏。
#include <deep_ep/common/ptx.cuh>        // release/acquire store、atomic add helper。

namespace deep_ep::elastic::handle {  // elastic device 端 NCCL/Gin handle 工具命名空间。
// Elastic/V2 设备端通信 handle 命名空间。

struct NCCLGin {  // 对 NCCL device-side Gin API 的统一封装。
    // NCCLGin 对 NCCL device-side Gin API 做轻封装，统一处理 LSA/NVLink 和 RDMA 路径。
#define IS_TEAM_WORLD(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld>) { code }  // 仅 world team 编译代码块。
// team_t 为 world team 时编译该代码块。
#define IS_TEAM_LSA(code) if constexpr (std::is_same_v<team_t, ncclTeamTagLsa>) { code }  // 仅 LSA team 编译代码块。
// team_t 为 LSA/NVLink team 时编译该代码块。
#define IS_TEAM_RAIL(code) if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) { code }  // 仅 Rail team 编译代码块。
// team_t 为 rail/RDMA team 时编译该代码块。
#define IS_TEAM_WORLD_RAIL(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld> or std::is_same_v<team_t, ncclTeamTagRail>) { code }  // world/rail 共享 RDMA Gin 路径。
// world 或 rail team 都支持 Gin RDMA API。
#define IS_TEAM_WORLD_LSA(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld> or std::is_same_v<team_t, ncclTeamTagLsa>) { code }  // world/LSA 可尝试 symmetric pointer。
// world 或 LSA team 都可能通过 LSA symmetric pointer 访问。
#define TEAM_WORLD_RAIL() ((std::is_same_v<team_t, ncclTeamTagWorld>) ? team_world : team_rail)  // 根据模板 team_t 选择 NCCL team 对象。
// 在 world/rail 二选一场景中返回对应 NCCL team 对象。

    const ncclDevComm_t& nccl_dev_comm;  // device communicator 引用。
    // 设备端 NCCL communicator 引用。
    const ncclWindow_t& nccl_window;  // symmetric window 引用。
    // symmetric memory window 引用。
    ncclGin gin;  // 具体 Gin context/QP。
    // NCCL Gin context wrapper，真正发 get/put/signal 的对象。
    ncclTeam team_world, team_lsa, team_rail;  // 三种 NCCL team 句柄。
    // 三种通信域：全局 world、本地 LSA/NVLink、跨 rail/RDMA。
    uint64_t lsa_base_ptr;  // 本 rank LSA window 基址。
    // 当前 rank 在 LSA window 中的基址，用于把裸指针转成 window offset。

    // TODO(NCCL): QP index should just be a hint or the users maintain the mapping?
    // NCCLGin 是对 NCCL device-side GIN API 的轻封装：同一份代码可以在 world
    // team、LSA/NVLink team、rail/RDMA team 间切换。qp_idx 选择具体 GIN context，
    // resource_sharing_mode 决定这个 context 在 CTA/GPU 间的共享粒度。
    __device__ __forceinline__
    NCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,  // communicator/window。
            const int& qp_idx = 0,  // 选择 Gin context/QP，下层 kernel 会根据 SM/channel 映射到不同 QP。
            const ncclGinResourceSharingMode& resource_sharing_mode = NCCL_GIN_RESOURCE_SHARING_GPU):  // QP 在 CTA/GPU 间的共享模式。
        nccl_dev_comm(nccl_dev_comm), nccl_window(nccl_window),  // 保存引用。
        // 保存 communicator 和 window 引用。
        gin(ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode)),  // 创建 Gin context。
        // 构造指定 QP/context 的 Gin 对象。
        team_world(ncclTeamWorld(nccl_dev_comm)), team_lsa(ncclTeamLsa(nccl_dev_comm)), team_rail(ncclTeamRail(nccl_dev_comm)) {  // 初始化三类 team。
        // 从 communicator 中取三种 team。
        // TODO: what if we only have 1 NVLink rank
        lsa_base_ptr = reinterpret_cast<uint64_t>(ncclGetLsaPointer(nccl_window, 0, team_lsa.rank));
        // 获取当前 LSA rank 的 window 基址；后续所有 ptr offset 都相对它计算。
    }

    template <typename team_t>  // team_t 决定 dst_rank_idx 属于 world/LSA/rail 哪个编号空间。
    __device__ __forceinline__ bool is_nvlink_accessible(const int& dst_rank_idx) const {  // 判断目标 rank 是否可直接 LSA 访问。
        // team_t 不同，dst_rank_idx 的语义也不同：
        // - LSA：rank 已在 NVLink 域内，必然可用 symmetric pointer
        // - World：先判断目标是否落在当前 rail 的 LSA rank 范围内
        // - Rail：只把同 rail rank 当作可直接访问，其他走 GIN RDMA
        IS_TEAM_LSA({
            // LSA team 内 rank 都在同一 LSA/NVLink 域，必然可直接访问。
            return true;
        })

        IS_TEAM_WORLD({
            // TODO(NCCL): optimize this function's cycles
            // return ncclTeamRankIsMember(team_lsa, team_world, dst_rank_idx);
            // 当前实现假设 world rank 按 [rail][lsa] 连续排列，因此可用除法区间判断本地 LSA 成员。
            return team_rail.rank * team_lsa.nRanks <= dst_rank_idx and
                   dst_rank_idx < (team_rail.rank + 1) * team_lsa.nRanks;
            // world rank 落在当前 rail 对应的 LSA rank 区间内时，可用 symmetric pointer。
        })

        IS_TEAM_RAIL({
            // TODO(NCCL): some ranks may be connected by NVLink, e.g., "2 + 2 + 4"
            return team_rail.rank == dst_rank_idx;
            // rail team 只把本 rail rank 视作本地可访问，其他 rank 走 RDMA。
        })
    }

    // ReSharper disable once CppNotAllPathsReturnValue
    template <typename dtype_t = void*>  // dtype_t 只影响指针类型，offset 始终按字节计算。
    __device__ __forceinline__
    uint64_t get_sym_offset(dtype_t* ptr) const {  // 将 window 内指针转成 LSA offset。
        // NCCL window 在每个 LSA peer 上地址不同，但 offset 相同；先转成相对本地
        // lsa_base_ptr 的 offset，再交给 ncclGetLsaPointer 得到远端 peer 指针。
        // 这个 offset 必须来自同一个 nccl_window 内的地址，不能传普通 cudaMalloc 指针。
        return reinterpret_cast<uint64_t>(ptr) - lsa_base_ptr;
        // 返回 ptr 相对 window 基址的偏移。
    }

    // ReSharper disable once CppNotAllPathsReturnValue
    template <typename team_t, typename dtype_t = void*>  // 返回类型与输入指针类型一致，便于调用方直接 LD/ST。
    __device__ __forceinline__
    dtype_t* get_sym_ptr(dtype_t* ptr, const int& dst_rank_idx) const {  // 返回目标 rank 的 symmetric pointer 或 nullptr。
        // 优先返回可直接 LD/ST 的 symmetric pointer；如果目标不在可见 NVLink 域内，
        // 返回 nullptr，调用方再降级到 GIN get/put/signal。
        IS_TEAM_RAIL({
            // rail team 语义下，只有本 rail rank 才能直接返回本地指针。
            return team_rail.rank == dst_rank_idx ? ptr : nullptr;
        })

        IS_TEAM_WORLD_LSA({
            // world/LSA team 尝试把本地 window offset 转成目标 LSA peer 指针。
            constexpr bool kIsTeamLSA = (std::is_same_v<team_t, ncclTeamTagLsa>);  // 当前模板是否 LSA team。
            // LSA team 的 dst_rank_idx 已经是 LSA rank；world team 需要换算。

            // Team world and not accessible by symmetric pointers
            if (not is_nvlink_accessible<team_t>(dst_rank_idx))
                // 不在本地 LSA/NVLink 域内时返回 nullptr，让调用方走 Gin。
                return nullptr;

            // Translate into NVLink rank index
            const auto dst_nvl_rank_idx = kIsTeamLSA ?  // 目标在 LSA team 内的 rank。
                dst_rank_idx : (dst_rank_idx - team_rail.rank * team_lsa.nRanks);
            // world rank 转成本 rail 内的 LSA rank。
            // 例如 rail.rank=2、LSA size=8、world dst=19，则 dst_nvl_rank_idx=3。

            // Local rank bypass
            // TODO(NCCL): support this
            if (dst_nvl_rank_idx == team_lsa.rank)
                // 目标就是本 rank，直接返回原指针。
                return ptr;

            // Get base ptr
            const auto dst_ptr = ncclGetLsaPointer(  // 通过 window offset 获取目标 LSA pointer。
                nccl_window, get_sym_offset(ptr), dst_nvl_rank_idx);
            // 用相同 offset 获取目标 LSA rank 的指针。
            // 返回值可直接用于 ld/st/TMA/atomic，但只在 LSA 可达 peer 上成立。
            return static_cast<dtype_t*>(dst_ptr);
            // 返回可直接 LD/ST/atomic 的远端 pointer。
        });
    }

    // NOTES: take care of this function when `team_t` is not LSA
    // Do not mix atomic add with gin signal into a single position
    template <typename team_t, typename dtype_t>  // dtype_t 当前在 RDMA signal 分支仅支持 64-bit，NVLink 分支可支持更多类型。
    __device__ __forceinline__
    void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,  // 目标计数地址、增量和目标 rank。
                     const int& extra_options = 0) const {  // 额外 Gin option，例如聚合请求。
        // 计数器类更新优先使用系统域 release atomic add，保证对端随后 acquire 读取
        // 能看到之前写入的数据；跨 RDMA 域时用 GIN VASignalAdd 模拟同样的计数语义。
        const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);  // 尝试获取直接可写指针。
        // 尝试获取可直接访问的 symmetric pointer。
        // Use symmetric pointers as much as possible, RDMA otherwise
        if (dst_ptr != nullptr) {
            // NVLink/本地路径使用系统域 release atomic add。
            // NOTES: local rank (or even NVLink-connected) for tag rail can also bypass
            // 直接 atomic add 的延迟低于 Gin signal，且 release 语义足以发布前序 payload。
            ptx::red_add_rel_sys(dst_ptr, value);
        } else {
            // 跨 RDMA 路径必须使用 Gin remote signal add。
            EP_DEVICE_ASSERT((not std::is_same_v<team_t, ncclTeamTagLsa>));
            // LSA team 理论上必须可 symmetric 访问，不应走到这里。
            EP_DEVICE_ASSERT((std::is_same_v<dtype_t, int64_t>) or (std::is_same_v<dtype_t, uint64_t>));
            // 当前 NCCL VASignalAdd 只支持 64-bit 计数语义。
            // TODO(NCCL): support all dtypes
            gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx,
                       // VASignalAdd 在远端 window offset 上执行 64-bit add，相当于远端 atomicAdd。
                       ncclGin_VASignalAdd(nccl_window, reinterpret_cast<int64_t>(sym_ptr) - lsa_base_ptr, static_cast<uint64_t>(value)),
                       ncclCoopThread(),
                       // 单线程发起 signal；调用方如需 warp 协作会在外层自行 elect。
                       ncclGin_None(),
                       // 无本地 completion action。
                       cuda::thread_scope_thread,
                       // 本地作用域只需覆盖发起线程。
                       cuda::thread_scope_device,
                       // 远端作用域覆盖目标 GPU device，使对端 acquire load 能看到 add。
                       ncclGinOptFlagsDefault | extra_options);
            // 在目标 rank 上对 window offset 执行 add。
            // extra_options 常用于聚合多个 Gin request，调用方需要之后 flush/wait。
        }
    }

    __device__ __forceinline__
    void wait(ncclGinRequest_t& request) const {  // 等待异步 Gin request 完成。
        // 等待 flush_async 返回的异步 Gin request 完成。
        // 调用方通常先批量 get/put，再 flush_async，再 wait，以减少每个小包的同步开销。
        gin.wait(request);
    }

    template <typename team_t, typename coop_t = ncclCoopThread, typename segment_t = ncclGin_SegmentDevice>
    // coop_t 决定一个 get 由 thread/warp/CTA 哪个粒度协作；segment_t 决定源/目标属于 device segment 还是其它 segment。
    __device__ __forceinline__
    void get(void* src_ptr, void* dst_ptr, const int& num_bytes, const int& src_rank_idx,  // 远端源、本地目标、字节数和源 rank。
             const int& extra_options = 0) const {  // 额外 Gin option。
        // GIN get/put 参数都使用 window offset，而不是裸指针。src/dst 指针都来自
        // 同一个 NCCL symmetric window，因此 reinterpret_cast 后减 lsa_base_ptr 即可。
        IS_TEAM_WORLD_RAIL({
            // GIN get 仅用于 world/rail team；LSA 应优先使用 symmetric pointer。
            // get 是远端读：从 src_rank_idx 的 src_ptr offset 读到本 rank 的 dst_ptr offset。
            gin.get(
                TEAM_WORLD_RAIL(),
                // 选择 world 或 rail team。
                src_rank_idx,
                // 源 rank。
                nccl_window, reinterpret_cast<int64_t>(src_ptr) - lsa_base_ptr,
                // 源 window offset。
                nccl_window, reinterpret_cast<int64_t>(dst_ptr) - lsa_base_ptr,
                // 目标 window offset。
                num_bytes,
                // 传输字节数。
                coop_t(),
                // 协作粒度，例如 thread/warp。
                ncclGin_None(),
                // 无额外 remote action。
                ncclGinOptFlagsDefault | extra_options,
                // 额外 option 可用于聚合请求。
                segment_t()
                // segment 类型，默认 device segment。
            );
        });
    }

    template <typename team_t, typename coop_t = ncclCoopThread>  // coop_t 需要和前面提交请求时的协作粒度匹配。
    __device__ __forceinline__
    void flush_async(const int& src_rank_idx, ncclGinRequest_t* request,  // 要 flush 的 rank 和输出 request。
                     const int& extra_options = 0) const {  // 额外 option。
        IS_TEAM_WORLD_RAIL({
            // flush_async 刷出此前对 src_rank_idx 聚合的 Gin 请求。
            // request 保存“这批请求”的完成句柄，wait(request) 后才能安全使用对应数据。
            gin.flushAsync(
                TEAM_WORLD_RAIL(),
                // world 或 rail team。
                src_rank_idx,
                // 要 flush 的目标/源 rank。
                request,
                // 输出异步 request，调用方稍后 wait。
                coop_t(),
                // 协作粒度。
                ncclGinOptFlagsDefault | extra_options
                // 额外 option。
            );
        });
    }

    template <typename team_t, typename remote_action_t>  // remote_action_t 可以是 SignalInc、VASignalAdd 等 NCCL Gin 远端动作。
    __device__ __forceinline__
    void signal(const int& dst_rank_idx, const remote_action_t& remote_action) const {  // 向目标 rank 发送远端动作。
        IS_TEAM_WORLD_RAIL({
            // 给目标 rank 发送一个 remote action，例如 signal increment。
            // signal 不携带 payload，常用于 PP 释放槽或 barrier 计数。
            gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx, remote_action);
        });
    }

    template <typename team_t,
              typename remote_action_t = ncclGin_None>  // 默认只 put 数据；传 remote_action 时可同时更新对端 signal。
    __device__ __forceinline__
	    void put(void* recv_sym_ptr, void* send_sym_ptr, const int& num_bytes, const int& dst_rank_idx,  // 远端接收、本地发送、字节数和目标 rank。
	             const int& extra_options = 0,  // 额外 Gin option。
	             const remote_action_t& remote_action = remote_action_t()) const {  // 可选远端动作。
	        // NOTES: local or NVLink put will also go through NIC via this API
	        // put 常和 remote_action 搭配：数据写到对端 recv_sym_ptr 后，顺手递增对端 signal，
	        // PP send/recv 就依赖这个动作把“槽位可读”事件传给接收端。
	        // recv_sym_ptr 和 send_sym_ptr 都必须属于同一个 nccl_window；Gin API 传输的不是
	        // CUDA 裸指针，而是相对 lsa_base_ptr 的 window offset。这样同一段代码在不同 rank
	        // 虚拟地址不一致时仍能定位到相同 symmetric allocation。
	        // 例：PP send 把本地 ring slot 写到 dst_rank 的 recv slot，并用 remote_action
	        // 增加对端 signal；接收端轮询 signal 后再从自己的 symmetric buffer 拷贝到用户 tensor。
	        IS_TEAM_WORLD_RAIL({
	            // put 当前只封装 world/rail RDMA 路径。
	            gin.put(TEAM_WORLD_RAIL(),
                    // 目标 team。
                    dst_rank_idx,
                    // 目标 rank。
                    // TODO: can we don't repeat the window?
                    // TODO: can we pass raw pointers?
                    nccl_window, reinterpret_cast<int64_t>(recv_sym_ptr) - lsa_base_ptr,
                    // 远端接收地址 offset。
                    nccl_window, reinterpret_cast<int64_t>(send_sym_ptr) - lsa_base_ptr,
                    // 本地发送地址 offset。
                    num_bytes,
                    // payload 字节数。
                    remote_action,
                    // put 完成后的远端动作。
                    ncclGin_None(),
                    // 无本地 action。
                    ncclCoopThread(),
                    // 单线程发起 put。
                    ncclGin_None(),
                    // 无 segment override。
	                    cuda::thread_scope_thread,
	                    // 本地线程作用域。
	                    cuda::thread_scope_device,
	                    // 远端设备作用域。
	                    ncclGinOptFlagsDefault | extra_options);
	            // option flags；extra_options 可打开 request aggregation 等队列行为。
	        });
	    }

    template <typename team_t, typename dtype_t>  // 写单个标量控制值，例如 tail、flag、prefix。
    __device__ __forceinline__
	    void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,  // 目标标量地址、值和目标 rank。
	                   const int& extra_options = 0) const {  // 额外 Gin option。
	        // put_value 写一个标量值，优先走 symmetric store，跨 RDMA 时走 Gin putValue。
	        // 它和 red_add_rel 的区别是“覆盖写”而不是“累加写”：适合 tail、prefix、ready flag
	        // 这类单生产者控制字段；若多个 writer 会同时更新同一槽位，应使用 atomic/reduction API。
	        const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);
        // 尝试获取可直接写的远端指针。
        if (dst_ptr != nullptr) {
            // NVLink/本地路径直接 system-scope relaxed store。
            // relaxed store 只适合写独立控制值；若作为发布信号，调用方应选择 release/add 版本。
            ptx::st_relaxed_sys(dst_ptr, value);
        } else {
            // 跨 RDMA 路径使用 Gin putValue。
            EP_DEVICE_ASSERT((not std::is_same_v<team_t, ncclTeamTagLsa>));
            // LSA team 不应走 RDMA 分支。
            gin.putValue(TEAM_WORLD_RAIL(),
                         dst_rank_idx,
                         // 目标 rank。
                         nccl_window, reinterpret_cast<int64_t>(sym_ptr) - lsa_base_ptr,
                         // 目标 window offset。
                         value,
                         // 要写入的标量。
                         ncclGin_None(),
                         ncclCoopThread(),
                         // 中文：Gin 参数里这里不使用额外 coop/group descriptor，只靠当前线程发起。
                         ncclGin_None(),
                         cuda::thread_scope_thread,
                         cuda::thread_scope_device,
                         // 中文：thread/device scope 分别描述发起线程与远端可见性范围。
                         ncclGinOptFlagsDefault | extra_options);
            // 发起 putValue。
            // putValue 常用于写 tail/prefix 这类小控制字段，比完整 put payload 更轻。
        }
    }

#undef IS_TEAM_WORLD
// 清理局部宏，避免污染其它 include。
#undef IS_TEAM_LSA
#undef IS_TEAM_RAIL
// 中文：WORLD/LSA/RAIL 判断宏只在 NCCLGin 类体内部使用，退出头文件前全部撤销。
#undef IS_TEAM_WORLD_RAIL
#undef IS_TEAM_WORLD_LSA
// 中文：TEAM_WORLD_RAIL 是组合 team 判断宏，最后撤销以避免影响后续头文件。
#undef TEAM_WORLD_RAIL
};

}  // namespace deep_ep::elastic::handle
// Elastic handle 命名空间结束。
