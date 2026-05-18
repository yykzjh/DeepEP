# 保存调用者原始目录，开发构建结束后切回。
original_dir=$(pwd)  # 保存原始工作目录，脚本结束时恢复。
# 解析脚本真实所在目录，避免从符号链接/相对路径启动时路径错位。
script_dir=$(realpath "$(dirname "$0")")  # 解析当前脚本所在的项目根目录。
# 切到项目根目录，setup.py 和 deep_ep 包路径都依赖这个位置。
cd "$script_dir"  # 进入项目根目录执行构建。

# 删除旧 build/dist 目录，避免加载过期的扩展产物。
rm -rf build dist  # 删除历史构建目录。
# 删除旧 egg-info，让开发构建重新生成包元数据。
rm -rf *.egg-info  # 删除历史 Python 包元数据。
# 执行本地 build，不打 wheel；产物会留在 build/ 下供软链接使用。
python setup.py build  # 执行开发构建，生成扩展 .so。

# 在 build 目录中找到编译出的 Python 扩展 .so 文件。
so_file=$(find build -name "*.so" -type f | head -n 1)  # 找到第一个构建出的扩展库。
# 如果找到了 .so，就把它软链接到 deep_ep 包目录，便于直接从源码树 import deep_ep。
if [ -n "$so_file" ]; then  # 找到扩展库时创建源码树软链接。
    # 链接目标使用 ../$so_file，因为链接文件位于 deep_ep/ 目录内。
    ln -sf "../$so_file" deep_ep/  # 将 .so 链接到 deep_ep 包目录下。
# 如果没有找到 .so，说明 build 失败或 setup.py 没有生成扩展。
else  # 没有 .so 表示构建失败或扩展未生成。
    # 输出错误到 stderr，便于 CI/终端区分普通日志和失败原因。
    echo "Error: No SO file found in build directory" >&2  # 输出错误信息到 stderr。
    # 非零退出，提醒调用者开发环境没有准备成功。
    exit 1  # 以失败状态退出。
fi

# 恢复调用者原始目录。
cd "$original_dir"  # 恢复调用者进入脚本前的目录。
