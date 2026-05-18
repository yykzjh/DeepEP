# 记录用户执行脚本时所在目录，脚本结束后会切回去，避免改变调用者 shell 的工作目录。
original_dir=$(pwd)  # 保存用户原始工作目录。
# 计算脚本所在的项目根目录；realpath 处理相对路径和符号链接，保证后续命令在仓库根执行。
script_dir=$(realpath "$(dirname "$0")")  # 解析脚本所在项目根目录。
# 切换到项目根目录，因为 setup.py、build/dist 等路径都按仓库根相对定位。
cd "$script_dir"  # 进入仓库根目录执行打包。

# 删除旧的 build/dist 产物，避免上一次构建残留影响当前 wheel 内容。
rm -rf build dist  # 删除历史 build/dist 产物。
# 删除旧的 egg-info 元数据，确保 setup.py 重新生成依赖和版本信息。
rm -rf *.egg-info  # 删除历史 egg-info 元数据。
# 构建 Python wheel；C++/CUDA 扩展会在 setup.py 内触发编译。
python setup.py bdist_wheel  # 构建 wheel 包。

# 回到用户原始目录，保持脚本对调用环境的副作用最小。
cd "$original_dir"  # 恢复用户原始工作目录。
