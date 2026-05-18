#pragma once  // 设备端数学 helper 被多个 JIT/legacy kernel include，防止重复定义。

namespace deep_ep::elastic::math {  // Elastic device/host 共用的轻量数学工具。

template <typename T>  // T 通常是 int/int64_t；要求支持 +、-、/，用于 token 数、字节数等整数维度。
__forceinline__ __device__ __host__ T ceil_div(T a, T b) {
    // 这里假设 a>=0 且 b>0；DeepEP 的容量、hidden、chunk 计算都满足这个前提。
    return (a + b - 1) / b;  // 整数向上除法，例如 ceil_div(10, 4)=3，常用于 block/token 分组数量。
}

template <typename T>  // constexpr 版本和运行期版本保持同一公式，避免模板 layout 与运行期 layout 不一致。
__forceinline__ __device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
    // 由于函数是 constexpr，调用方可以把返回值用于数组长度、模板默认参数和 static_assert。
    return (a + b - 1) / b;  // constexpr 版本可用于模板参数、静态数组大小等编译期表达式。
}

template <typename T, bool kDoCeilAlignment = true>  // kDoCeilAlignment 编译期选择向上/向下对齐，避免运行期分支。
__forceinline__ __device__ __host__ T align(T a, T b) {
    // kDoCeilAlignment=true 表示向上对齐，false 表示向下截断到 b 的倍数；
    // buffer layout 中两种情况都会用到，例如计算最大容量和计算已用空间。
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T, bool kDoCeilAlignment = true>  // 编译期对齐 helper，常和 constexpr_ceil_div 一起参与静态 layout 计算。
__forceinline__ __device__ __host__ constexpr T constexpr_align(T a, T b) {
    // 例：constexpr_align<int, true>(13, 8)=16；constexpr_align<int, false>(13, 8)=8。
    return (kDoCeilAlignment ? constexpr_ceil_div(a, b) : (a / b)) * b;  // constexpr 对齐，供模板编译期 layout 计算使用。
}

template <typename dtype_t>  // dtype_t 可为 int/uint64_t 等计数器类型；只依赖和 0 比较。
__forceinline__ __device__ __host__ bool is_decoded_positive_ready(const dtype_t& value) {
    // 一些 workspace 计数器用“负数编码未 ready，非负数表示 ready”的协议。
    // 读取方先判断是否可用，再用 encode_decode_positive 还原真实正数。
    return value >= 0;
}

template <typename dtype_t>  // 和 is_decoded_positive_ready 配套，用同一个公式完成编码与解码。
__forceinline__ __device__ __host__ dtype_t encode_decode_positive(const dtype_t& value) {
    // 自反编码：encode/decode 都是 -value-1。例：真实值 7 编成 -8；
    // 对 -8 再调用一次得到 7。这样 0 也能被编码成 -1，不和 ready 状态混淆。
    return -value - static_cast<dtype_t>(1);
}

template <typename dtype_t = void>  // 返回类型默认 void*，调用方也可指定 int8_t/int4/float 等 typed pointer。
__forceinline__ __device__ __host__ dtype_t* advance_ptr(void* ptr, const int64_t num_bytes) {
    // 对 void* 做字节级偏移的统一 helper，避免 C++ 对 void* 不能直接加减带来的
    // 重复 reinterpret_cast。模板参数决定返回指针类型。
    return reinterpret_cast<dtype_t*>(static_cast<int8_t*>(ptr) + num_bytes);
}

__forceinline__ __device__ __host__ ptrdiff_t ptr_diff(const void* ptr, const void* base) {
    // 两个指针应来自同一 workspace/buffer；跨 allocation 相减在 C++ 语义上没有可移植意义。
    return static_cast<const int8_t*>(ptr) - static_cast<const int8_t*>(base);  // 返回字节级偏移，常用于检查 workspace 内指针布局。
}

template <typename dtype_a_t, typename dtype_b_t>  // dtype_b_t 是两个 dtype_a_t 的位级容器，例如两个 half 打到一个 uint32。
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    // 把两个小类型按内存布局打包成一个大类型，常用于把两个 BF16/half lane
    // 合成一个向量寄存器。要求 sizeof(dtype_b)==2*sizeof(dtype_a)。
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;  // 打包后的大类型变量，未初始化内容会被下面两个小 lane 覆盖。
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);  // 把 packed 的内存视作两个 dtype_a_t 槽。
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;  // 第 0 个 lane 放 x，第 1 个 lane 放 y。
    return packed;  // 返回按内存位模式打包后的值，不做数值转换。
}

template <typename dtype_a_t, typename dtype_b_t>  // 返回 tuple 的版本适合 `auto [x, y] = unpack2<...>(packed)`。
__device__ __forceinline__ std::tuple<dtype_a_t, dtype_a_t> unpack2(const dtype_b_t& packed) {
    // pack2 的反操作，按小类型数组视图读取，不做数值转换。
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);  // 把 packed 的内存视作两个 dtype_a_t 槽。
    dtype_a_t x = unpacked_ptr[0], y = unpacked_ptr[1];  // 分别读取两个 lane。
    return {x, y};  // 以 tuple 返回，适合调用方结构化绑定。
}

template <typename dtype_a_t, typename dtype_b_t>  // 引用输出版本适合性能敏感路径，避免 tuple 临时对象。
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");  // 编译期检查打包/解包类型大小匹配。
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);                 // 按小类型视图解释 packed 内存。
    x = unpacked_ptr[0], y = unpacked_ptr[1];                                        // 通过引用返回两个 lane，避免构造 tuple。
}

}  // namespace deep_ep::elastic::math
