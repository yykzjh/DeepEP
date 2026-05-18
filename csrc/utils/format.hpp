#pragma once  // 防止本头文件在同一翻译单元中被重复包含。

// fmt 只以 header-only 方式使用，避免 PyTorch extension 链接时额外处理 libfmt。
// 其他 C++ 文件 include 本 wrapper 即可获得 fmt::format。
#define FMT_HEADER_ONLY  // 告诉 fmt 库以内联/header-only 方式展开实现。
#include <fmt/base.h>    // 引入 fmt 的基础类型和轻量格式化声明。
#include <fmt/format.h>  // 引入 fmt::format 等完整格式化 API。
