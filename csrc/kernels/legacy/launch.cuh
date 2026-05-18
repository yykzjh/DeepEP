#pragma once  // legacy kernel launch/template switch helper，避免多个 .cu 重复定义宏。

#include "compiled.cuh"  // 引入 DISABLE_SM90_FEATURES、LEGACY_NUM_MAX_NVL_PEERS 等编译期常量。

#ifndef SETUP_LAUNCH_CONFIG
#ifndef DISABLE_SM90_FEATURES
// legacy kernel 仍使用运行时模板实例化而非 elastic JIT。SM90 路径通过
// cudaLaunchKernelEx 打开 cooperative launch，并在偶数 SM 时设置 clusterDim=2，
// 以匹配 TMA/cluster 相关优化。
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                       \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; /* grid=num_sms，block=num_threads，shared=0，stream=调用方传入。 */ \
    cudaLaunchAttribute attr[2];                                                /* 两个 launch attribute：cooperative 与 cluster dimension。 */ \
    attr[0].id = cudaLaunchAttributeCooperative;                                /* 第 0 个属性声明 cooperative launch。 */ \
    attr[0].val.cooperative = 1;                                                /* 打开 cooperative，允许 kernel 内做更强同步假设。 */ \
    attr[1].id = cudaLaunchAttributeClusterDimension;                           /* 第 1 个属性声明 CTA cluster 维度。 */ \
    attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);                      /* 偶数 SM 用 x=2，把成对 block 放进同一 cluster。 */ \
    attr[1].val.clusterDim.y = 1;                                               /* y 维不分 cluster。 */ \
    attr[1].val.clusterDim.z = 1;                                               /* z 维不分 cluster。 */ \
    cfg.attrs = attr;                                                           /* 把属性数组挂到 launch config。 */ \
    cfg.numAttrs = 2                                                            /* launch config 中共有两个属性。 */
#else
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
    int __num_sms = (sms);                        /* 非 SM90 回退路径保存 grid 大小。 */ \
    int __num_threads = (threads);                /* 非 SM90 回退路径保存 block 大小。 */ \
    auto __stream = (stream)                      /* 非 SM90 回退路径保存 CUDA stream。 */
#endif
#endif

#ifndef LAUNCH_KERNEL
#ifndef DISABLE_SM90_FEATURES
// LAUNCH_KERNEL 隐藏 SM90 与旧架构的启动差异：SM90 走 cudaLaunchKernelEx，
// 非 SM90 退回普通 <<<grid, block>>> 并立即检查 launch error。
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))  /* SM90 路径通过扩展 launch config 启动 kernel。 */
#else
#define LAUNCH_KERNEL(config, kernel, ...)                                                 \
    do {                                                                                   /* 用 do-while 包住多语句宏，调用处可像普通语句一样加分号。 */ \
        kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__);                    /* 旧架构使用普通 CUDA triple-chevron launch。 */ \
        cudaError_t e = cudaGetLastError();                                                /* 立即读取 launch error，避免错误延迟到后续 API。 */ \
        if (e != cudaSuccess) {                                                            /* launch 失败时构造 DeepEP 异常。 */ \
            EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); /* 记录文件、行号和 CUDA 错误字符串。 */ \
            fprintf(stderr, "%s\n", cuda_exception.what());                                /* 同时写 stderr，方便多进程日志定位。 */ \
            throw cuda_exception;                                                          /* 抛给 C++/Python 调用层。 */ \
        }                                                                                  /* 结束错误处理分支。 */ \
    } while (0)                                                                            /* 宏语句结束。 */
#endif
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#ifndef DISABLE_SM90_FEATURES
// 使用 TMA 的 kernel 需要显式提高 dynamic shared memory 上限；smem_size 由调用
// 位置计算，宏写回 cfg.dynamicSmemBytes。
#define SET_SHARED_MEMORY_FOR_TMA(kernel)                                                                                \
    EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); /* 提高 kernel 动态 shared memory 上限。 */ \
    cfg.dynamicSmemBytes = smem_size;                                                                                    /* launch config 同步设置实际 dynamic smem 字节数。 */
#else
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()  /* 非 SM90/TMA 路径无需设置 dynamic smem。 */
#endif
#endif

#define SWITCH_RANKS(case_macro)                           \
    /* 根据 rank 数选择编译期模板实例，支持 2/4/8 卡 NVLink 域。 */       \
    switch (num_ranks) {                                   /* 按运行时 num_ranks 选择编译期模板参数。 */ \
        case 2:                                            /* 2 卡 NVLink 域。 */ \
            case_macro(2);                                 /* 展开调用方传入的模板实例化宏。 */ \
        case 4:                                            /* 4 卡 NVLink 域。 */ \
            case_macro(4);                                 /* 展开 4-rank 实例。 */ \
        case 8:                                            /* 8 卡 NVLink 域，DeepEP legacy 默认节点内规模。 */ \
            case_macro(8);                                 /* 展开 8-rank 实例。 */ \
        default:                                           /* 其它规模没有预编译实例。 */ \
            EP_HOST_ASSERT(false and "Unsupported ranks"); /* 直接报错，避免落入错误模板。 */ \
    }                                                      /* switch 结束。 */ \
    while (false)                                          /* 与 case_macro 内部 break 配合的 legacy 写法。 */

#define SWITCH_RDMA_RANKS(case_macro)                           \
    /* internode legacy 路径按 RDMA rank 数实例化，num_ranks/8 表示节点数。 */ \
    switch (num_ranks / LEGACY_NUM_MAX_NVL_PEERS) {             /* 全局 rank 数除以每节点 8 卡，得到 RDMA/node rank 数。 */ \
        case 2:                                                 /* 2 节点实例。 */ \
            case_macro(2);                                      /* 展开 2-rdma-rank 模板。 */ \
        case 3:                                                 /* 3 节点实例。 */ \
            case_macro(3);                                      /* 展开 3-rdma-rank 模板。 */ \
        case 4:                                                 /* 4 节点实例。 */ \
            case_macro(4);                                      /* 展开 4-rdma-rank 模板。 */ \
        case 6:                                                 /* 6 节点实例。 */ \
            case_macro(6);                                      /* 展开 6-rdma-rank 模板。 */ \
        case 8:                                                 /* 8 节点实例。 */ \
            case_macro(8);                                      /* 展开 8-rdma-rank 模板。 */ \
        case 12:                                                /* 12 节点实例。 */ \
            case_macro(12);                                     /* 展开 12-rdma-rank 模板。 */ \
        case 16:                                                /* 16 节点实例。 */ \
            case_macro(16);                                     /* 展开 16-rdma-rank 模板。 */ \
        case 18:                                                /* 18 节点实例。 */ \
            case_macro(18);                                     /* 展开 18-rdma-rank 模板。 */ \
        case 20:                                                /* 20 节点实例。 */ \
            case_macro(20);                                     /* 展开 20-rdma-rank 模板。 */ \
        default:                                                /* 未列出的节点数没有预编译实例。 */ \
            EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); /* 报错提示调用配置不支持。 */ \
    }                                                           /* switch 结束。 */ \
    while (false)                                               /* legacy 宏统一的尾部占位。 */

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)         \
    switch (num_ranks) {                                   /* 同时按 dtype 和 rank 数选择模板实例。 */ \
        case 2:                                            /* 2 rank 实例。 */ \
            case_macro(dtype, 2);                          /* 把 dtype 与 2 传给调用方宏。 */ \
        case 4:                                            /* 4 rank 实例。 */ \
            case_macro(dtype, 4);                          /* 把 dtype 与 4 传给调用方宏。 */ \
        case 8:                                            /* 8 rank 实例。 */ \
            case_macro(dtype, 8);                          /* 把 dtype 与 8 传给调用方宏。 */ \
        default:                                           /* 其它 rank 数不支持。 */ \
            EP_HOST_ASSERT(false and "Unsupported ranks"); /* 报错。 */ \
    }                                                      /* switch 结束。 */ \
    while (false)                                          /* 宏尾部占位。 */

#define SWITCH_TYPES(case_macro)                          \
    switch (type) {                                       /* 按 cudaDataType_t 选择 C++ dtype。 */ \
        case CUDA_R_16BF:                                 /* BF16 类型。 */ \
            case_macro(nv_bfloat16);                      /* 展开 nv_bfloat16 模板实例。 */ \
        default:                                          /* legacy 目前没有其它 dtype 预编译实例。 */ \
            EP_HOST_ASSERT(false and "Unsupported type"); /* 报错。 */ \
    }                                                     /* switch 结束。 */ \
    while (false)                                         /* 宏尾部占位。 */

#define SWITCH_HIDDEN(case_macro)                           \
    /* hidden 必须是预编译白名单中的 MoE hidden size，保证内层向量化展开可在编译期确定。 */ \
    switch (hidden) {                                       /* 按 hidden 选择编译期常量，便于循环展开和向量化。 */ \
        case 2048:                                          /* hidden=2048。 */ \
            case_macro(2048);                               /* 展开 2048 模板实例。 */ \
        case 2560:                                          /* hidden=2560。 */ \
            case_macro(2560);                               /* 展开 2560 模板实例。 */ \
        case 3072:                                          /* hidden=3072。 */ \
            case_macro(3072); /* for gpt-oss */             /* 展开 GPT-OSS 相关实例。 */ \
        case 4096:                                          /* hidden=4096。 */ \
            case_macro(4096);                               /* 展开 4096 模板实例。 */ \
        case 5120:                                          /* hidden=5120。 */ \
            case_macro(5120);                               /* 展开 5120 模板实例。 */ \
        case 6144:                                          /* hidden=6144。 */ \
            case_macro(6144); /* For qwen3 coder */         /* 展开 Qwen3 coder 相关实例。 */ \
        case 7168:                                          /* hidden=7168，DeepSeek MoE 常见配置。 */ \
            case_macro(7168);                               /* 展开 7168 模板实例。 */ \
        case 8192:                                          /* hidden=8192。 */ \
            case_macro(8192);                               /* 展开 8192 模板实例。 */ \
        default:                                            /* 不在白名单内的 hidden 不支持 legacy 预编译。 */ \
            EP_HOST_ASSERT(false and "Unsupported hidden"); /* 报错。 */ \
    }                                                       /* switch 结束。 */ \
    while (false)                                           /* 宏尾部占位。 */
