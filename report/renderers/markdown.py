"""
Markdown 格式报告渲染器模块

将 ExplainabilityReport 渲染为结构化的 Markdown 格式文本，
支持执行摘要、详细分析、技术附录和图表引用。
"""

from __future__ import annotations

from typing import Any

from core.models import Chart, ExplainabilityReport


class MarkdownRenderer:
    """Markdown 格式报告渲染器

    将可解释性报告的各部分渲染为 Markdown 格式，
    保持清晰的层次结构和可读性。
    """

    def render(self, report: ExplainabilityReport) -> str:
        """渲染为 Markdown 格式

        按照报告结构依次渲染标题、元信息、执行摘要、
        详细分析、技术附录和图表列表。

        Args:
            report: 可解释性分析报告

        Returns:
            str: Markdown 格式的报告文本
        """
        sections: list[str] = []

        # 报告标题和元信息
        sections.append(self._render_header(report))

        # 目录
        sections.append(self._render_toc())

        # 执行摘要
        if report.executive_summary:
            sections.append(self._render_executive_summary(report))

        # 详细分析
        if report.detailed_analysis:
            sections.append(self._render_detailed_analysis(report))

        # 技术附录
        if report.technical_appendix:
            sections.append(self._render_technical_appendix(report))

        # 图表列表
        if report.charts:
            sections.append(self._render_charts(report))

        return "\n\n".join(sections)

    def _render_header(self, report: ExplainabilityReport) -> str:
        """渲染报告标题和元信息

        Args:
            report: 可解释性分析报告

        Returns:
            str: Markdown 格式的标题区域
        """
        lines: list[str] = []

        # 标题
        title = report.metadata.title if report.metadata else "可解释性分析报告"
        lines.append(f"# {title}")
        lines.append("")

        # 元信息表格
        if report.metadata:
            lines.append("| 属性 | 值 |")
            lines.append("|------|-----|")

            if report.metadata.generated_at:
                lines.append(f"| 生成时间 | {report.metadata.generated_at} |")
            if report.metadata.domain:
                lines.append(f"| 业务领域 | {report.metadata.domain} |")
            if report.metadata.data_source:
                lines.append(f"| 数据来源 | {report.metadata.data_source} |")

            if report.metadata.analysis_config:
                config = report.metadata.analysis_config
                lines.append(f"| 目标受众 | {config.audience} |")
                lines.append(f"| 分析深度 | {config.depth} |")
                if config.target_variable:
                    lines.append(f"| 目标变量 | {config.target_variable} |")
                lines.append(f"| 因果分析 | {'启用' if config.causal_enabled else '禁用'} |")
                lines.append(f"| 预测模拟 | {'启用' if config.predictive_enabled else '禁用'} |")

        return "\n".join(lines)

    def _render_toc(self) -> str:
        """渲染目录

        Returns:
            str: Markdown 格式的目录
        """
        return (
            "## 目录\n\n"
            "1. [执行摘要](#执行摘要)\n"
            "2. [详细分析](#详细分析)\n"
            "3. [技术附录](#技术附录)\n"
            "4. [图表列表](#图表列表)"
        )

    def _render_executive_summary(self, report: ExplainabilityReport) -> str:
        """渲染执行摘要章节

        Args:
            report: 可解释性分析报告

        Returns:
            str: Markdown 格式的执行摘要
        """
        lines: list[str] = ["## 执行摘要", ""]
        lines.append(report.executive_summary)
        return "\n".join(lines)

    def _render_detailed_analysis(self, report: ExplainabilityReport) -> str:
        """渲染详细分析章节

        Args:
            report: 可解释性分析报告

        Returns:
            str: Markdown 格式的详细分析
        """
        lines: list[str] = ["## 详细分析", ""]
        lines.append(report.detailed_analysis)
        return "\n".join(lines)

    def _render_technical_appendix(self, report: ExplainabilityReport) -> str:
        """渲染技术附录章节

        Args:
            report: 可解释性分析报告

        Returns:
            str: Markdown 格式的技术附录
        """
        lines: list[str] = ["## 技术附录", ""]
        lines.append(report.technical_appendix)
        return "\n".join(lines)

    def _render_charts(self, report: ExplainabilityReport) -> str:
        """渲染图表列表章节

        以表格形式列出报告中包含的所有图表及其描述。

        Args:
            report: 可解释性分析报告

        Returns:
            str: Markdown 格式的图表列表
        """
        lines: list[str] = ["## 图表列表", ""]
        lines.append("| 序号 | 图表类型 | 标题 | 描述 |")
        lines.append("|------|---------|------|------|")

        for i, chart in enumerate(report.charts, 1):
            type_map = {
                "bar": "柱状图",
                "line": "折线图",
                "scatter": "散点图",
                "heatmap": "热力图",
                "graph": "关系图",
            }
            type_str = type_map.get(chart.type, chart.type)
            lines.append(f"| {i} | {type_str} | {chart.title} | {chart.description} |")

        return "\n".join(lines)

    @staticmethod
    def render_chart_data(chart: Chart) -> str:
        """将单个图表数据渲染为 Markdown 表格或列表

        Args:
            chart: 图表定义

        Returns:
            str: Markdown 格式的图表数据表示
        """
        lines: list[str] = [f"### {chart.title}", ""]
        lines.append(f"*{chart.description}*")
        lines.append("")

        if chart.type == "bar":
            labels = chart.data.get("labels", [])
            values = chart.data.get("values", [])
            lines.append("| 变量 | 值 |")
            lines.append("|------|-----|")
            for label, value in zip(labels, values):
                lines.append(f"| {label} | {value} |")

        elif chart.type == "heatmap":
            variables = chart.data.get("variables", [])
            matrix = chart.data.get("matrix", [])
            if variables and matrix:
                header = "| | " + " | ".join(variables) + " |"
                separator = "|---|" + "|".join(["---"] * len(variables)) + "|"
                lines.append(header)
                lines.append(separator)
                for var, row in zip(variables, matrix):
                    row_str = " | ".join(f"{v:.2f}" for v in row)
                    lines.append(f"| {var} | {row_str} |")

        elif chart.type == "graph":
            edges = chart.data.get("edges", [])
            lines.append("| 来源 | 目标 | 效应大小 | p值 |")
            lines.append("|------|------|---------|-----|")
            for edge in edges:
                lines.append(
                    f"| {edge['from']} | {edge['to']} | "
                    f"{edge['effect_size']} | {edge['p_value']} |"
                )

        else:
            # 通用渲染：直接输出 data 字典
            for key, value in chart.data.items():
                lines.append(f"**{key}**: {value}")

        return "\n".join(lines)
