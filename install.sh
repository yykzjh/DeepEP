# 保存调用者原始目录，安装结束后恢复。
original_dir=$(pwd)  # 保存调用脚本前所在目录。
# 获取脚本所在目录；这里保持原写法，不强制 realpath，兼容简单相对路径调用。
script_dir=$(dirname "$0")  # 获取脚本所在目录，作为项目根目录。
# 切到项目根目录，确保 setup.py 和 dist 路径解析正确。
cd "$script_dir"  # 进入项目根目录。

# 删除旧 dist，避免 pip install dist/*.whl 时匹配到多个历史 wheel。
rm -rf dist  # 清理旧 wheel 输出目录。
# 先构建 wheel；编译扩展和打包 Python 代码都由 setup.py 完成。
python setup.py bdist_wheel  # 重新构建 wheel。
# 安装刚生成的 wheel 到当前 Python 环境。
pip install dist/*.whl  # 安装刚生成的 wheel。

# 恢复用户进入脚本前的目录。
cd "$original_dir"  # 恢复用户原始工作目录。
