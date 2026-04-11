"""
报告渲染器模块

提供 Markdown 和 HTML 两种格式的报告渲染器。
"""

from report.renderers.html import HTMLRenderer
from report.renderers.markdown import MarkdownRenderer

__all__ = [
    "MarkdownRenderer",
    "HTMLRenderer",
]
