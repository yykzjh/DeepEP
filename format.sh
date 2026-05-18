#!/usr/bin/env bash  # 使用 bash 运行，脚本依赖数组、[[ ]] 等 bash 语法。
# 本脚本用于在提交前统一格式化 Python 与 C/C++/CUDA 代码。
# Usage:
# 中文：下面示例说明先开发、再对相对 main 变化的文件执行格式化。
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
# 中文：提交示例只说明推荐流程，本脚本本身不会自动 git add/commit。
#
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# 中文：脚本依赖 git merge-base 计算变化范围，避免全仓库无关文件被改动。
# You are encouraged to run this locally before pushing changes for review.

# 任意命令失败时立即退出；pipefail 保证管道中间命令失败也能传递失败状态。
set -eo pipefail  # -e 遇错退出，-o pipefail 让管道任一阶段失败都会使整条管道失败。

# 如果 yapf/ruff 不存在，就按 requirements-lint.txt 安装项目指定版本。
if ! (yapf --version &>/dev/null && ruff --version &>/dev/null); then  # 同时检查 yapf 和 ruff 是否可执行。
    pip install -r requirements-lint.txt  # 缺少任一工具时安装项目锁定的 lint 依赖。
fi  # 结束 lint 工具安装检查。

# 读取当前 yapf 版本，后续目前没有强制校验，但保留变量便于调试。
YAPF_VERSION=$(yapf --version | awk '{print $2}')  # 提取 yapf 版本号，例如 0.40.2。
# 读取当前 ruff 版本，后续目前没有强制校验，但保留变量便于调试。
RUFF_VERSION=$(ruff --version | awk '{print $2}')  # 提取 ruff 版本号，例如 0.x.y。

# 输出 yapf 阶段开始标记。
echo 'yapf: Check Start'  # 打印 yapf 阶段开始。

# yapf 递归并行格式化 Python 文件。
YAPF_FLAGS=(  # yapf 通用参数数组，避免字符串拼接导致空格转义问题。
    '--recursive'  # 递归搜索 Python 文件。
    '--parallel'  # 并行格式化，加快全仓处理。
)  # yapf flags 数组结束。

# 排除 build 目录，避免格式化生成产物。
YAPF_EXCLUDES=(  # yapf 排除规则数组。
    '--exclude' 'build/**'  # build 是生成目录，不应被格式化。
)  # yapf excludes 数组结束。

# Format specified files
# 格式化调用者传入的指定文件列表。
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"  # 对传入文件原地执行 yapf。
}

# Format all files
# 格式化仓库内所有未排除 Python 文件。
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .  # 对当前仓库全量 Python 文件原地格式化。
}

# Format files that differ from main branch
# 只格式化相对 main/origin/main 发生变化的 Python 文件。
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # 中文：ruff 没有文件参数时会从 STDIN 读入，这在自动脚本里会表现为卡住。
    # waiting for STDIN.
    #
    # 中文：这里显式判断文件列表非空，避免 ruff 在没有路径参数时从标准输入等待。
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # 中文：ACM 只包含新增、复制、修改文件，跳过删除文件以免 formatter 报路径不存在。
    # exist on both branches.
    # 优先使用 origin/main；没有远端分支时退回本地 main。
    if git show-ref --verify --quiet refs/remotes/origin/main; then  # 如果存在远端 main 引用。
        BASE_BRANCH="origin/main"  # 优先使用远端 main 作为 diff 基线。
    else
        BASE_BRANCH="main"  # 没有远端引用时使用本地 main。
    fi  # 结束基线分支选择。

    # merge-base 是当前分支与主线的共同祖先，用于得到本分支改动范围。
    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"  # 找到当前 HEAD 与基线分支的共同祖先。

    # 只有存在新增/复制/修改的 Python 文件时才调用 yapf，避免 xargs 空输入。
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then  # 存在新增/复制/修改的 Python 文件才继续。
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"  # 并行把变更文件交给 yapf。
    fi  # 没有 Python diff 时跳过 yapf，避免 xargs 空输入。
}

# If `--all` is passed, then any further arguments are ignored and the
# entire python directory is formatted.
if [[ "$1" == '--all' ]]; then  # 用户传 --all 时执行全量格式化。
   # --all 表示忽略 diff 范围，格式化全部 Python 文件。
   format_all  # 调用全量 Python 格式化。
else
   # Format only the files that changed in last commit.
   # 默认只格式化当前分支变更的 Python 文件，减少无关 churn。
   format_changed  # 只格式化相对主线变更的 Python 文件。
fi  # yapf 范围选择结束。
# 输出 yapf 阶段结束标记。
echo 'yapf: Done'  # 打印 yapf 阶段结束。

# 输出 ruff 阶段开始标记。
echo 'ruff: Check Start'  # 打印 ruff 阶段开始。
# Lint specified files
# 对调用者传入的文件执行 ruff check。
lint() {
    ruff check "$@"  # 对传入路径执行 ruff 静态检查。
}

# Lint files that differ from main branch. Ignores dirs that are not slated
# for autolint yet.
# 只 lint 相对 main/origin/main 发生变化的 Python 文件。
lint_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # 中文：lint 阶段同样需要这个保护，避免 CI 或本地终端等待输入。
    # waiting for STDIN.
    #
    # 中文：lint_changed 与 format_changed 使用同一保护逻辑，只处理新增/复制/修改且仍存在的文件。
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # 中文：这里和 format_changed 保持一致，避免 lint 阶段读取已删除路径。
    # exist on both branches.
    # 优先使用远端 origin/main，缺失时使用本地 main。
    if git show-ref --verify --quiet refs/remotes/origin/main; then  # 如果远端 main 存在。
        BASE_BRANCH="origin/main"  # 使用 origin/main 作为 lint diff 基线。
    else
        BASE_BRANCH="main"  # 否则退回本地 main。
    fi  # 结束基线选择。

    # 计算当前分支和主线的共同祖先。
    MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"  # 当前分支与主线的共同祖先。

    # 只有存在 Python 改动时才运行 ruff，避免 ruff 等待 stdin。
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then  # 存在 Python 文件变更才运行 ruff。
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             ruff check  # 将变更 Python 文件传给 ruff check。
    fi  # 无 Python 变更时跳过。
}

# Run Ruff
# If `--all` is passed, then any further arguments are ignored and the
# entire python directory is linted.
if [[ "$1" == '--all' ]]; then  # --all 时 lint 整个仓库。
   # --all 下 lint 整个仓库。
   lint .   # 对当前目录执行 ruff check。
else
   # Check only the files that changed in last commit.
   # 默认只检查当前分支变更的 Python 文件。
   lint_changed  # 只 lint 变更 Python 文件。
fi  # ruff 范围选择结束。

# 输出 ruff 阶段结束标记。
echo 'ruff: Done'  # 打印 ruff 阶段结束。

# # params: tool name, tool version, required version
# 检查工具版本是否匹配 requirements-lint.txt，不匹配时重新安装。
tool_version_check() {
    # $2 是当前版本，$3 是要求版本。
    if [[ $2 != $3 ]]; then  # 当前版本和要求版本不一致。
        # 输出版本不匹配提示。
        echo "Wrong $1 version installed: $3 is required, not $2."  # 告诉用户实际版本和期望版本。
        # 重新安装 lint 依赖，尽量恢复到项目要求版本。
        pip install -r requirements-lint.txt  # 重新安装 requirements-lint 中的工具版本。
    fi  # 版本一致时不做任何事。
}

# 输出 clang-format 阶段开始标记。
echo 'clang-format: Check Start'  # 打印 clang-format 阶段开始。
# If clang-format is available, run it; otherwise, skip
# 只有系统能找到 clang-format 时才格式化 C/C++/CUDA 文件。
if command -v clang-format &>/dev/null; then  # 只有系统 PATH 中能找到 clang-format 才执行 C/C++ 格式化。
    # clang-format --version 输出中第三列通常是版本号。
    CLANG_FORMAT_VERSION=$(clang-format --version | awk '{print $3}')  # 从 clang-format --version 提取版本号。
    # 对照 requirements-lint.txt 中声明的 clang-format 版本。
    tool_version_check "clang-format" "$CLANG_FORMAT_VERSION" "$(grep clang-format requirements-lint.txt | cut -d'=' -f3)"  # 对照 requirements-lint 要求版本。

    # clang-format -i 表示原地修改文件。
    CLANG_FORMAT_FLAGS=("-i")  # clang-format -i 表示原地修改文件。

    # Format all C/C++ files in the repo, excluding specified directories
    # 格式化仓库内全部 C/C++/CUDA 文件。
    clang_format_all() {
        # Replace "#pragma unroll" by "// #pragma unroll"
        # clang-format 会调整 pragma 附近格式，先临时注释掉 #pragma unroll。
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
            -not -path "./build/*" \
            -exec perl -pi -e 's/#pragma unroll/\/\/#pragma unroll/g' {} +  # 对所有 C/C++/CUDA 文件临时注释 pragma。
        # 对所有 C/C++/CUDA 源文件运行 clang-format。
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
            -not -path "./build/*" \
            -exec clang-format -i {} +  # 对所有匹配源文件运行 clang-format -i。
        # Replace "// #pragma unroll" by "#pragma unroll"
        # 格式化后恢复 #pragma unroll，保证 CUDA 编译器仍看到展开提示。
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
            -not -path "./build/*" \
            -exec perl -pi -e 's/\/\/ *#pragma unroll/#pragma unroll/g' {} +  # 恢复被临时注释的 pragma unroll。
    }

    # Format changed C/C++ files relative to main
    # 只格式化相对主线变更的 C/C++/CUDA 文件。
    clang_format_changed() {
        # 优先使用 origin/main，缺失时退回 main。
        if git show-ref --verify --quiet refs/remotes/origin/main; then  # 如果存在 origin/main。
            BASE_BRANCH="origin/main"  # 使用远端 main 作为 C/C++ diff 基线。
        else
            BASE_BRANCH="main"  # 否则使用本地 main。
        fi  # 结束基线选择。

        # 计算当前分支与主线的共同祖先。
        MERGEBASE="$(git merge-base $BASE_BRANCH HEAD)"  # 计算当前分支和基线的共同祖先。

        # 只有存在 C/C++/CUDA 改动时才运行格式化。
        if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' &>/dev/null; then  # 有 C/C++/CUDA 文件变更才运行。
            # 临时注释 #pragma unroll，避免 clang-format 误处理。
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' | xargs perl -pi -e 's/#pragma unroll/\/\/#pragma unroll/g'  # 仅对变更源文件临时注释 pragma。
            # 对变更文件原地 clang-format。
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' | xargs clang-format -i  # 仅格式化变更源文件。
            # 恢复 #pragma unroll。
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' | xargs perl -pi -e 's/\/\/ *#pragma unroll/#pragma unroll/g'  # 恢复变更源文件中的 pragma。
        fi  # 无 C/C++/CUDA 变更时跳过。
    }

    # 根据 --all 选择全量格式化或仅格式化变更文件。
    if [[ "$1" == '--all' ]]; then  # --all 时格式化全部 C/C++/CUDA 文件。
       # If --all is given, format all eligible C/C++ files
       clang_format_all  # 调用全量 clang-format。
    else
       # Otherwise, format only changed C/C++ files
       clang_format_changed  # 调用变更文件 clang-format。
    fi  # clang-format 范围选择结束。
else
    # 没有安装 clang-format 时跳过 C/C++ 格式化，Python 阶段仍然有效。
    echo "clang-format not found. Skipping C/C++ formatting."  # 提示用户缺少 clang-format。
fi  # clang-format 可用性分支结束。
# 输出 clang-format 阶段结束标记。
echo 'clang-format: Done'  # 打印 clang-format 阶段结束。

# Check if there are any uncommitted changes after all formatting steps.
# If there are, ask the user to review and stage them.
# 如果格式化产生了 diff，就打印文件列表和 diff，并以失败退出提醒开发者处理。
if ! git diff --quiet &>/dev/null; then  # 如果格式化导致工作区出现 diff。
    # 提示格式化已经修改文件。
    echo 'Reformatted files. Please review and stage the changes.'  # 提醒用户检查格式化产生的改动。
    # 打印未暂存文件标题。
    echo 'Changes not staged for commit:'  # 打印未暂存变更标题。
    echo  # 空行。
    # 打印变更文件名。
    git --no-pager diff --name-only  # 输出变更文件列表。

    # 打印完整 diff，方便 CI 日志中直接查看格式化结果。
    echo 'You can also copy-paste the diff below to fix the lint:'  # 在 CI 日志中提示可复制 diff。
    echo  # 空行。
    git --no-pager diff  # 输出完整 diff。

    # 非零退出，让调用者知道需要处理格式化产生的改动。
    exit 1  # 以失败退出，迫使调用方处理格式化 diff。
fi  # 没有 diff 则继续。

# 没有格式化 diff，说明所有检查通过。
echo 'All checks passed'  # 所有格式化/lint 检查通过且没有产生新 diff。
