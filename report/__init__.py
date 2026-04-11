"""
报告生成模块 —— 模板渲染与多格式输出

提供报告生成器和多种格式的渲染器，将分析结果转化为
结构化的可解释性报告。
"""

from report.generator import ReportGenerator
from report.renderers import HTMLRenderer, MarkdownRenderer

__all__ = [
    "ReportGenerator",
    "MarkdownRenderer",
    "HTMLRenderer",
]
