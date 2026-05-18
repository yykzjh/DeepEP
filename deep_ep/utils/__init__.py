"""DeepEP 工具子包的兼容入口。

目前只重导出 EventHandle，保留该文件是为了让历史代码可以继续使用
``deep_ep.utils.EventHandle`` 形式访问事件句柄。
"""

# For forward compatibility  # 兼容旧版导入路径，避免用户代码因模块拆分而失效。
# noinspection PyUnresolvedReferences  # 静态检查器可能看不到 re-export 语义，这里显式抑制误报。
from .event import EventHandle  # 兼容旧代码：允许用户从 deep_ep.utils 直接导入 EventHandle。
