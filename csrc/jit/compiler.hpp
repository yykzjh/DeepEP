#pragma once  // Compiler 是 header-only JIT 组件，防止重复定义静态成员和类。

#include <fcntl.h>       // open/fsync/close 用于分布式文件系统上的 cache 持久化。
#include <filesystem>   // JIT cache、临时目录、cubin/ptx/sass 路径都用 filesystem 表达。
#include <fstream>      // 写 kernel.cu/kernel.ptx 等中间文件。
#include <nvrtc.h>      // 预留 NVRTC 后端类型；当前默认实现使用 NVCCCompiler。
#include <optional>     // compile 可选输出 PTX 文件路径。
#include <regex>        // 解析 nvcc --version 输出以及检查 ptxas local memory。
#include <string>       // compiler flags、kernel source、signature 都用 std::string。

#include <deep_ep/common/exception.cuh>  // EP_HOST_ASSERT/NCCL/CUDA 检查宏。

#include "../utils/format.hpp"       // fmt::format 轻量格式化命令和 cache key。
#include "../utils/hash.hpp"         // get_hex_digest/get_uuid/make_dirs/safe_remove_all 等工具。
#include "../utils/lazy_init.hpp"    // LazyInit 延迟创建全局 compiler。
#include "../utils/system.hpp"       // get_env/call_external_command 调用环境变量和外部命令。
#include "cache.hpp"                 // kernel_runtime_cache 负责编译后 module 加载复用。
#include "device_runtime.hpp"        // device_runtime 提供当前 GPU 架构信息。

namespace deep_ep::jit {
// JIT 编译子系统命名空间。

class Compiler {
    // Compiler 抽象出“根据源码生成 cubin 并写入 cache”的公共流程；
    // 具体后端由 NVCCCompiler::compile 实现。
public:
    static std::filesystem::path library_root_path;
    // deep_ep Python 包根目录，例如 .../site-packages/deep_ep。
    static std::filesystem::path library_include_path;
    // JIT include 根目录，通常是 library_root_path/include。
    static std::filesystem::path cuda_home;
    // CUDA_HOME，由 Python 探测后传入。
    static std::filesystem::path nccl_root;
    // NCCL 安装根目录，用于 -I nccl_root/include。
    static std::filesystem::path cuobjdump_path;
    // cuobjdump 路径，用于可选 SASS 反汇编。

    static void prepare_init(const std::string& library_root_path,  // Python 包根目录。
                             const std::string& cuda_home_path_by_python,  // Python 探测到的 CUDA_HOME。
                             const std::string& nccl_root_path_by_python) {  // Python 探测到的 NCCL 根目录。
        // 由 deep_ep/__init__.py 调用。三个路径决定 JIT 编译是否能复现安装环境：
        // - library_root_path：安装后的 deep_ep 包目录，下面必须有 include/deep_ep；
        // - cuda_home：提供 NVCC/NVRTC/cuobjdump；
        // - nccl_root：提供 NCCL host/device 头文件，特别是 Gin 相关 device 类型。
        // NOTES: if you are adding some third-party includes for kernels, please add its hash value
        Compiler::library_root_path = library_root_path;
        // 保存 Python 传入的包根目录。
        Compiler::library_include_path = Compiler::library_root_path / "include";
        // JIT 生成源码中的 #include <deep_ep/...> 都从这里解析。
        Compiler::cuda_home = cuda_home_path_by_python;
        // 保存 CUDA 安装路径。
        Compiler::nccl_root = nccl_root_path_by_python;
        // 保存 NCCL 安装路径。
        Compiler::cuobjdump_path = Compiler::cuda_home / "bin" / "cuobjdump";
        // cuobjdump 与 nvcc 一般同处 CUDA_HOME/bin。
    }

    std::string signature, flags;
    // signature 描述编译器版本，flags 描述命令行选项；二者都进入 cache key。
    std::filesystem::path cache_dir_path;
    // JIT cache 根目录，默认 ~/.deep_ep，也可由 EP_JIT_CACHE_DIR 覆盖。

    Compiler() {
        // 构造时要求 init_jit 已经设置所有路径。
        EP_HOST_ASSERT(not library_root_path.empty());
        // 缺少包根目录时无法找到 include。
        EP_HOST_ASSERT(not library_include_path.empty());
        // 缺少 include 根目录时 JIT 源码无法编译。
        EP_HOST_ASSERT(not cuda_home.empty());
        // 缺少 CUDA_HOME 时无法找到 nvcc/cuobjdump。
        EP_HOST_ASSERT(not nccl_root.empty());
        // 缺少 NCCL root 时无法 include nccl_device 头。
        EP_HOST_ASSERT(not cuobjdump_path.empty());
        // cuobjdump 路径必须可构造，实际存在性在使用时由命令检查。

        // Cache settings
        // 默认缓存位于 ~/.deep_ep；集群环境通常通过 EP_JIT_CACHE_DIR 指到共享/本地高速盘。
        // build() 中的原子 rename 逻辑就是为了多个 rank 同时编译同一个 kernel 时不会互相破坏。
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_ep";
        // 默认 cache 跟随用户 HOME，便于跨进程复用。
        if (const auto env_cache_dir_path = get_env<std::string>("EP_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            // 用户可把 cache 放到本地 NVMe 或共享文件系统。
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        // 子类会改成如 NVCC12.4 的真实签名。
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940,3012 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("EP_JIT_CPP_STANDARD", 20));
        // 默认 C++20；diag-suppress 关闭 CUDA 模板/系统头中常见的无关警告。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PTXAS_VERBOSE", 0))
            // 调试或 ptxas verbose 时打开 ptxas 详细寄存器/栈使用输出。
            flags += " --ptxas-options=--verbose";
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_WITH_LINEINFO", 0))
            // lineinfo 让 Nsight/SASS 能关联到生成的 kernel.cu 行号。
            flags += " -Xcompiler -rdynamic -lineinfo";
        if (get_env("EP_GIN_GDAKI_DEBUG", 0))
            // 开启 NCCL Gin GDAKI device 侧调试宏。
            flags += " -DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1";
        flags += fmt::format(" -I {}/include", nccl_root.c_str());
        // 把 NCCL include 加入搜索路径，JIT kernel 需要 nccl_device 头。

        // Some special flags for EP
        // TODO: make it more general, e.g. `EP_JIT_EXTRA_FLAGS`
        if (int num_topk_idx_bits = get_env("EP_NUM_TOPK_IDX_BITS", 0); num_topk_idx_bits != 0)
            // topk_idx_t 位宽可通过宏覆盖，必须进入 flags/cache key 防止 ABI 混用。
            flags += fmt::format(" -DEP_NUM_TOPK_IDX_BITS={}", num_topk_idx_bits);
    }

    virtual ~Compiler() = default;
    // 抽象基类虚析构，允许通过 shared_ptr<Compiler> 销毁子类。

    std::filesystem::path make_tmp_dir() const {  // 创建/返回 JIT cache 下的临时目录根。
        // 所有临时编译目录都放在 cache_dir/tmp 下；build() 会再追加 UUID，
        // 避免同一节点多个 rank 同时编译时写到同一路径。
        return make_dirs(cache_dir_path / "tmp");
    }

    static void fsync_path(const std::filesystem::path& path) {  // 对单个文件或目录执行 fsync。
        // 分布式文件系统上 close() 不一定立刻让其它节点看到文件内容；
        // fsync 文件或目录可以降低其它 rank 读取半成品 cubin 的概率。
        const auto fd = ::open(path.c_str(), O_RDONLY);
        // 以只读方式打开文件或目录，只为拿到可 fsync 的 fd。
        if (fd >= 0) {
            // 打开失败时忽略：某些文件系统/权限可能不支持目录 fsync。
            ::fsync(fd);
            // 强制把文件内容或目录项刷到文件系统。
            ::close(fd);
            // 关闭 fd，避免泄漏。
        }
    }

    // Recursively fsync a directory: files and subdirectories first (bottom-up), then the directory itself
    // NOTES: ensures data and directory entries are visible on other nodes in distributed filesystems
    static void fsync_dir(const std::filesystem::path& dir_path) { // NOLINT(*-no-recursion)  // 递归 fsync 一个目录树。
        // 递归 fsync cache 临时目录，确保 rename 前所有文件内容和目录项都可见。
        for (const auto& entry: std::filesystem::directory_iterator(dir_path)) {
            // 遍历目录下所有文件/子目录。
            if (entry.is_directory())
                // 子目录先递归刷盘。
                fsync_dir(entry.path());
            else if (entry.is_regular_file())
                // 普通文件直接 fsync。
                fsync_path(entry.path());
        }
        fsync_path(dir_path);
        // 最后 fsync 目录本身，确保目录项可见。
    }

    static void put(const std::filesystem::path& path, const std::string& data) {  // 以 binary 模式写文件并 fsync。
        // 写 kernel.cu/kernel.ptx 等中间文件的统一入口。binary 模式避免换行转换
        // 影响源码 hash 或 nvcc 报错行号。
        std::ofstream out(path, std::ios::binary);
        // binary 模式写文件，避免不同平台换行转换影响源码内容。
        EP_HOST_ASSERT(out.write(data.data(), data.size()));
        // 写入完整 data，失败直接报错。
        out.close();
        // close 后再 fsync，确保文件句柄刷新。

        // NOTES: fsync to ensure the data is visible to other processes (e.g., NVCC)
        // on distributed filesystems, where `close()` alone does not guarantee persistence
        fsync_path(path);
        // 在分布式文件系统上让后续 NVCC/其他 rank 能看到完整内容。
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {  // 编译或加载指定 JIT kernel。
        // 一个 kernel 的缓存 key 不只包含模板实例代码，还包含 compiler signature 和 flags。
        // 例：同一个 dispatch，在 hidden=7168/topk=8/EP=16 与 hidden=4096/topk=4/EP=8
        // 下会生成不同 code；即使 code 相同，EP_NUM_TOPK_IDX_BITS 或 NCCL include 路径变化也会
        // 通过 flags/signature 进入 key，避免 ABI/类型不匹配。
        const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
        // 用分隔符拼接所有影响 cubin 的因素，作为 hash 输入。
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));
        // 最终 cache 目录形如 ~/.deep_ep/cache/kernel.dispatch.<hash>。

        // 先查进程内 runtime cache；命中时连磁盘目录检查都不必重复做。
        // KernelRuntime 持有已加载的 CUDA module/kernel handle。
        if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // cache miss 时编译到 tmp_dir_path。最终 cache 目录名已经由源码/flags hash
        // 决定，所以不同 rank 同时 miss 会编译出相同内容，再竞争 rename。
        // NOTES: renaming a directory is atomic on both local and distributed filesystems,
        // avoiding the stale inode issue that occurs when renaming individual files
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        // 每个编译尝试使用唯一临时目录，避免多 rank 写同一目录。
        make_dirs(tmp_dir_path);
        // 创建临时目录。

        // compile() 由 NVCCCompiler 或 NVRTCCompiler 实现。dump PTX/ASM 开关只影响
        // 是否额外保留调试产物，不影响最终 kernel.cubin 的加载路径。
        const auto tmp_cubin_path = tmp_dir_path / "kernel.cubin";
        // 最终供 KernelRuntime 加载的 cubin 路径。
        if (get_env<int>("EP_JIT_DUMP_ASM") or get_env<int>("EP_JIT_DUMP_PTX")) {
            // 需要 PTX 或 ASM dump 时额外要求 compile 生成 PTX。
            const auto tmp_ptx_path = tmp_dir_path / "kernel.ptx";
            // PTX 调试输出路径。
            compile(code, tmp_dir_path, tmp_cubin_path, tmp_ptx_path);
            // 编译 cubin，并额外写 PTX。
        } else {
            // 常规路径只生成 cubin，减少编译时间和磁盘 IO。
            compile(code, tmp_dir_path, tmp_cubin_path);
        }

        // SASS dump 用于性能分析，例如确认 TMA、LDG.E.NA、warpgroup 指令是否按预期生成。
        if (get_env<int>("EP_JIT_DUMP_ASM") or get_env<int>("EP_JIT_DUMP_SASS")) {
            // 需要 SASS 时调用 cuobjdump 反汇编 cubin。
            const auto tmp_sass_path = tmp_dir_path / "kernel.sass";
            // SASS 输出路径。
            disassemble(tmp_cubin_path, tmp_sass_path);
            // 生成 kernel.sass。
        }

        // Fsync before rename to ensure visibility on distributed filesystems
        fsync_dir(tmp_dir_path);
        // rename 前先递归刷盘，降低其它 rank 看到半成品目录的风险。

        // Atomically rename the temporary directory to the final cache path
        // NOTES: if another rank already created dir_path, rename will fail — that's fine
        make_dirs(dir_path.parent_path());
        // 确保 cache 父目录存在。
        std::error_code error_code;
        // 使用 error_code 版本避免异常打断多 rank 竞争逻辑。
        std::filesystem::rename(tmp_dir_path, dir_path, error_code);
        // 原子把临时目录移动到最终 cache 目录。
        if (error_code) {
            // rename 失败通常表示另一个 rank 已经率先创建了相同 hash 的目录。
            // Another rank beat us, then clean up our dir and use the existing one
            // 中文：多进程 JIT 会竞争同一个 hash 目录，失败者只需清理自己的临时目录。
            // NOTES: avoid `std::filesystem::remove_all` here — it can segfault on
            // 中文：这里使用 safe_remove_all，是为了规避分布式文件系统目录项过期导致的崩溃。
            // distributed filesystems, when concurrent processes operate
            // on the same parent directory, causing stale directory entries
            safe_remove_all(tmp_dir_path);
            // 清理本 rank 的临时目录，保留已经存在的最终 cache。
        }

        // 不管当前 rank 是 rename 成功还是输给其它 rank，最终都从 dir_path 加载。
        // 如果 dir_path 仍无效，kernel_runtime_cache->get 会返回 nullptr 并触发断言。
        const auto runtime = kernel_runtime_cache->get(dir_path);
        // 从最终 cache 目录加载/复用 KernelRuntime。
        EP_HOST_ASSERT(runtime != nullptr);
        // 如果仍然无法加载，说明编译或多 rank cache 形成过程异常。
        return runtime;
        // 返回可 launch 的 runtime。
    }

    static void disassemble(const std::filesystem::path &cubin_path, const std::filesystem::path &sass_path) {  // 用 cuobjdump 生成 SASS。
        // cuobjdump 是外部进程，输出重定向到 kernel.sass。失败时直接断言，因为
        // dump 开关通常用于开发诊断，静默失败会误导后续分析。
        const auto command = fmt::format("{} --dump-sass {} > {}", cuobjdump_path.c_str(), cubin_path.c_str(), sass_path.c_str());
        // cuobjdump 命令把 SASS stdout 重定向到 sass_path。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
            // 调试模式打印完整命令，便于用户复现。
            printf("Running cuobjdump command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        // 执行外部命令并收集返回码/stdout+stderr。
        if (return_code != 0) {
            // cuobjdump 失败时输出错误文本。
            printf("cuobjdump failed: %s\n", output.c_str());
            EP_HOST_ASSERT(false and "cuobjdump failed");
            // dump 失败通常说明 CUDA_HOME/cuobjdump 或 cubin 异常。
        }
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::optional<std::filesystem::path> &ptx_path = std::nullopt) const = 0;  // 子类输出 cubin，可选输出 PTX。
    // 子类实现真正编译；dir_path 是临时目录，cubin_path 是必须输出的 cubin。
};

EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
// 在 header-only 类中声明/定义 static path 成员。
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
// static include path 成员定义。
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
// static CUDA_HOME 成员定义。
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, nccl_root);
// static NCCL root 成员定义。
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuobjdump_path);
// static cuobjdump path 成员定义。

class NVCCCompiler final: public Compiler {
    // 当前唯一实际编译器后端：通过外部 nvcc 生成 cubin。
    std::filesystem::path nvcc_path;
    // nvcc 可执行文件路径，可由 EP_JIT_NVCC_COMPILER 覆盖。

    std::pair<int, int> get_nvcc_version() const {  // 调用 nvcc --version 并解析 major/minor。
        // 解析 nvcc 版本，既用于兼容性检查，也进入 compiler signature。
        EP_HOST_ASSERT(std::filesystem::exists(nvcc_path));
        // nvcc 路径必须存在。

        // nvcc --version 输出中包含 "release 12.x"，用正则解析 major/minor。
        // DeepEP JIT 依赖 CUDA 12.3+ 的 device 侧 NCCL/Gin 与 Hopper 特性。
        const auto command = std::string(nvcc_path) + " --version";
        // 调用 nvcc --version 获取版本字符串。
        const auto [return_code, output] = call_external_command(command);
        // 执行命令并收集输出。
        EP_HOST_ASSERT(return_code == 0);
        // nvcc 必须可执行且正常返回。

        // The version should be at least 12.3
        int major, minor;
        // 解析出的主/次版本。
        std::smatch match;
        // 正则匹配结果。
        EP_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        // 输出中必须包含 release X.Y。
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        // 把 X.Y 解析成整数。
        EP_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        // DeepEP JIT 依赖 CUDA 12.3+ 特性。
        return {major, minor};
        // 返回版本用于 signature 和架构后缀判断。
    }

public:
    NVCCCompiler() {  // 构造 NVCC 后端，确定 nvcc 路径、目标架构和编译 flags。
        // 允许 EP_JIT_NVCC_COMPILER 指向自定义 nvcc，例如容器内多 CUDA 版本时
        // 显式选择和运行时 driver 兼容的编译器。
        nvcc_path = cuda_home / "bin" / "nvcc";
        // 默认使用 CUDA_HOME/bin/nvcc。
        cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        // 同步设置 cuobjdump 路径。
        if (const auto env_nvcc_path = get_env<std::string>("EP_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            // 用户可指定自定义 nvcc，例如容器内多版本 CUDA。
            nvcc_path = env_nvcc_path;
        const auto [nvcc_major, nvcc_minor] = get_nvcc_version();
        // 获取并校验 nvcc 版本。
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);
        // 编译器版本进入 cache key，避免不同 nvcc 复用不兼容 cubin。

        // get_arch(false, allow_family_suffix) 会返回例如 90a 或 100f 这样的目标架构。
        // CUDA 12.9+ 才支持 family suffix，因此旧 nvcc 只用数字架构，避免 nvcc 报错。
        const auto arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        // 根据当前 GPU 和 nvcc 能力选择 sm_90a/sm_100f 等目标架构。
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            // 中文：CUDA relaxed constexpr/extended lambda 是模板化 kernel 代码所需编译选项。
                            "-O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), arch);
        // 追加 DeepEP include、目标架构、优化和 CUDA 扩展 lambda/constexpr 支持。
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,  // JIT 源码与工作目录。
                 const std::filesystem::path &cubin_path,  // 输出 cubin 路径。
                 const std::optional<std::filesystem::path> &ptx_path) const override {  // 可选 PTX 输出路径。
        // kernel.cu 会保存在 cache 目录，便于用户查看 JIT 生成的真实 CUDA 源码。
        // include hash 已经由 LaunchRuntime 拼到源码顶部。
        const auto code_path = dir_path / "kernel.cu";
        // JIT 生成源码保存在临时目录中，最终随 cache 一起保留。
        put(code_path, code);
        // 写入 kernel.cu 并 fsync。

        // 编译时 cwd 设为临时空目录，避免用户当前工作目录里的同名头文件意外遮蔽
        // C++ 标准库或 DeepEP include；输出直接生成 cubin。
        const auto compile_dir = make_tmp_dir();
        // 编译 cwd 使用独立临时目录，避免当前工作目录污染 include 解析。
        const auto command = fmt::format("cd {} && {} {} -cubin -o {} {}",
            compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        // 生成 cubin 的 nvcc 命令。
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
            // 调试时打印完整 nvcc 命令。
            printf("Running NVCC command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        // 执行 nvcc 并收集编译输出。
        if (return_code != 0) {
            // 编译失败时把 nvcc 输出打印给用户。
            printf("NVCC compilation failed: %s\n", output.c_str());
            EP_HOST_ASSERT(false and "NVCC compilation failed");
            // 直接断言，避免后续加载不存在/半成品 cubin。
        }

        // Compile to PTX if needed
        if (ptx_path.has_value()) {
            // 用户要求 dump PTX 时再跑一次 -ptx 编译。
            const auto ptx_command = fmt::format("cd {} && {} {} -ptx -o {} {}",
                compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), ptx_path->c_str(), flags);
            // 生成 PTX 的 nvcc 命令。
            if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
                // 打印 PTX 命令便于复现。
                printf("Running NVCC PTX command: %s\n", ptx_command.c_str());
            const auto [ptx_return_code, ptx_output] = call_external_command(ptx_command);
            // 执行 PTX 编译。
            if (ptx_return_code != 0) {
                // PTX 生成失败同样视为 JIT 失败。
                printf("NVCC PTX compilation failed: %s\n", ptx_output.c_str());
                EP_HOST_ASSERT(false and "NVCC PTX compilation failed");
                // 避免留下不完整调试产物。
            }
        }

        // Check local memory usage
        if (get_env("EP_JIT_PTXAS_CHECK", 0))
            // CI/调优时可要求 ptxas 不使用 local memory，防止寄存器溢出影响性能。
            EP_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PTXAS_VERBOSE", 0))
            // 打印 ptxas verbose 输出，包括寄存器、shared/local memory 使用。
            printf("%s", output.c_str());
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    // 全局 compiler 延迟构造：只有首次 JIT kernel build 时才检查 nvcc 和设备架构。
    return std::make_shared<NVCCCompiler>();
    // 当前默认后端为 NVCCCompiler。
});

} // namespace deep_ep::jit
// DeepEP JIT 子系统命名空间结束。
