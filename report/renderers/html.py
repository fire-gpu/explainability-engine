"""
HTML 格式报告渲染器模块

将 ExplainabilityReport 渲染为带内联 CSS 样式的 HTML 页面，
支持打印友好输出和响应式布局。
"""

from __future__ import annotations

from typing import Any

from core.models import Chart, ExplainabilityReport


class HTMLRenderer:
    """HTML 格式报告渲染器，带基础样式

    生成自包含的 HTML 文档，内联 CSS 样式，无需外部依赖。
    支持打印友好模式，可通过浏览器直接打印为 PDF。
    """

    def render(self, report: ExplainabilityReport) -> str:
        """渲染为 HTML 格式

        生成完整的 HTML 文档，包含内联 CSS 样式，
        按照报告结构渲染各章节内容。

        Args:
            report: 可解释性分析报告

        Returns:
            str: 完整的 HTML 文档字符串
        """
        sections: list[str] = []

        # 报告标题和元信息
        sections.append(self._render_header(report))

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

        body_content = "\n".join(sections)

        return self._wrap_html(body_content)

    def _wrap_html(self, body_content: str) -> str:
        """将内容包裹为完整的 HTML 文档

        包含 DOCTYPE、head（内联 CSS）和 body 结构。

        Args:
            body_content: HTML 正文内容

        Returns:
            str: 完整的 HTML 文档
        """
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>可解释性分析报告</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        {body_content}
    </div>
</body>
</html>"""

    @staticmethod
    def _get_css() -> str:
        """获取内联 CSS 样式

        包含基础排版、章节样式、表格样式和打印优化。

        Returns:
            str: CSS 样式字符串
        """
        return """
        /* 基础排版 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei",
                         sans-serif;
            line-height: 1.8;
            color: #333;
            background-color: #f8f9fa;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: #fff;
            padding: 40px 50px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border-radius: 4px;
        }

        /* 标题 */
        h1 {
            font-size: 28px;
            color: #1a1a2e;
            border-bottom: 3px solid #4361ee;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }

        h2 {
            font-size: 22px;
            color: #16213e;
            margin-top: 36px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        }

        h3 {
            font-size: 18px;
            color: #0f3460;
            margin-top: 24px;
            margin-bottom: 12px;
        }

        /* 段落 */
        p {
            margin-bottom: 12px;
            text-align: justify;
        }

        /* 列表 */
        ul, ol {
            margin-left: 24px;
            margin-bottom: 12px;
        }

        li {
            margin-bottom: 6px;
        }

        /* 元信息表格 */
        .meta-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 24px;
            font-size: 14px;
        }

        .meta-table th {
            background-color: #4361ee;
            color: #fff;
            padding: 8px 12px;
            text-align: left;
            font-weight: 500;
        }

        .meta-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }

        .meta-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        /* 数据表格 */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }

        table th {
            background-color: #e8edf5;
            color: #1a1a2e;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            border: 1px solid #ddd;
        }

        table td {
            padding: 8px 12px;
            border: 1px solid #ddd;
        }

        table tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        table tr:hover {
            background-color: #e8edf5;
        }

        /* 执行摘要高亮框 */
        .executive-summary {
            background-color: #f0f4ff;
            border-left: 4px solid #4361ee;
            padding: 20px 24px;
            margin: 20px 0;
            border-radius: 0 4px 4px 0;
        }

        /* 技术附录 */
        .technical-appendix {
            background-color: #f9f9f9;
            padding: 20px 24px;
            margin: 20px 0;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            font-size: 14px;
        }

        /* 图表卡片 */
        .chart-card {
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 16px;
            margin: 12px 0;
            background-color: #fff;
        }

        .chart-card h4 {
            color: #0f3460;
            margin-bottom: 8px;
        }

        .chart-card .chart-desc {
            color: #666;
            font-size: 13px;
            font-style: italic;
            margin-bottom: 12px;
        }

        /* 强调文本 */
        strong {
            color: #1a1a2e;
        }

        /* 代码块 */
        code {
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 13px;
        }

        /* 分隔线 */
        hr {
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 32px 0;
        }

        /* 打印优化 */
        @media print {
            body {
                background-color: #fff;
                padding: 0;
            }

            .container {
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }

            h2 {
                page-break-before: auto;
            }

            .executive-summary,
            .technical-appendix {
                break-inside: avoid;
            }

            table {
                break-inside: avoid;
            }
        }

        /* 响应式 */
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            h1 {
                font-size: 24px;
            }

            h2 {
                font-size: 20px;
            }

            table {
                font-size: 12px;
            }
        }
        """

    def _render_header(self, report: ExplainabilityReport) -> str:
        """渲染报告标题和元信息

        Args:
            report: 可解释性分析报告

        Returns:
            str: HTML 格式的标题区域
        """
        title = report.metadata.title if report.metadata else "可解释性分析报告"
        lines: list[str] = [f"<h1>{self._escape(title)}</h1>"]

        if report.metadata:
            lines.append('<table class="meta-table">')
            lines.append("<thead><tr><th>属性</th><th>值</th></tr></thead>")
            lines.append("<tbody>")

            meta_rows = [
                ("生成时间", report.metadata.generated_at or ""),
                ("业务领域", report.metadata.domain or ""),
                ("数据来源", report.metadata.data_source or ""),
            ]

            if report.metadata.analysis_config:
                config = report.metadata.analysis_config
                meta_rows.append(("目标受众", config.audience))
                meta_rows.append(("分析深度", config.depth))
                if config.target_variable:
                    meta_rows.append(("目标变量", config.target_variable))
                meta_rows.append(("因果分析", "启用" if config.causal_enabled else "禁用"))
                meta_rows.append(("预测模拟", "启用" if config.predictive_enabled else "禁用"))

            for label, value in meta_rows:
                lines.append(
                    f"<tr><td>{self._escape(label)}</td>"
                    f"<td>{self._escape(value)}</td></tr>"
                )

            lines.append("</tbody></table>")

        return "\n".join(lines)

    def _render_executive_summary(self, report: ExplainabilityReport) -> str:
        """渲染执行摘要章节

        Args:
            report: 可解释性分析报告

        Returns:
            str: HTML 格式的执行摘要
        """
        content = self._text_to_html(report.executive_summary)
        return (
            f'<h2 id="executive-summary">执行摘要</h2>\n'
            f'<div class="executive-summary">\n'
            f"{content}\n"
            f"</div>"
        )

    def _render_detailed_analysis(self, report: ExplainabilityReport) -> str:
        """渲染详细分析章节

        Args:
            report: 可解释性分析报告

        Returns:
            str: HTML 格式的详细分析
        """
        content = self._text_to_html(report.detailed_analysis)
        return (
            f'<h2 id="detailed-analysis">详细分析</h2>\n'
            f"{content}"
        )

    def _render_technical_appendix(self, report: ExplainabilityReport) -> str:
        """渲染技术附录章节

        Args:
            report: 可解释性分析报告

        Returns:
            str: HTML 格式的技术附录
        """
        content = self._text_to_html(report.technical_appendix)
        return (
            f'<h2 id="technical-appendix">技术附录</h2>\n'
            f'<div class="technical-appendix">\n'
            f"{content}\n"
            f"</div>"
        )

    def _render_charts(self, report: ExplainabilityReport) -> str:
        """渲染图表列表章节

        以卡片形式展示所有图表的元信息。

        Args:
            report: 可解释性分析报告

        Returns:
            str: HTML 格式的图表列表
        """
        lines: list[str] = ['<h2 id="charts">图表列表</h2>']

        type_map = {
            "bar": "柱状图",
            "line": "折线图",
            "scatter": "散点图",
            "heatmap": "热力图",
            "graph": "关系图",
        }

        for chart in report.charts:
            type_str = type_map.get(chart.type, chart.type)
            lines.append(
                f'<div class="chart-card">\n'
                f"  <h4>{self._escape(chart.title)}</h4>\n"
                f'  <p class="chart-desc">'
                f"[{type_str}] {self._escape(chart.description)}</p>\n"
                f"</div>"
            )

        return "\n".join(lines)

    @staticmethod
    def _escape(text: str) -> str:
        """HTML 特殊字符转义

        Args:
            text: 原始文本

        Returns:
            str: 转义后的安全文本
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _text_to_html(text: str) -> str:
        """将纯文本转换为 HTML 段落

        将文本按空行分段，每段包裹在 <p> 标签中。
        保留 Markdown 风格的 **粗体** 标记并转换为 <strong>。

        Args:
            text: 纯文本内容

        Returns:
            str: HTML 格式的内容
        """
        # 先处理粗体标记
        import re

        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

        # 按空行分段
        paragraphs = text.split("\n\n")
        html_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 处理列表项（以 - 或数字. 开头的行）
            lines = para.split("\n")
            if all(line.strip().startswith(("- ", f"{i}. ")) for i, line in enumerate(lines, 1) if line.strip()):
                # 有序列表
                html_lines = ["<ol>"]
                for line in lines:
                    line = line.strip()
                    if line.startswith("- "):
                        html_lines.append(f"<li>{line[2:]}</li>")
                    else:
                        # 去掉数字前缀
                        cleaned = re.sub(r'^\d+\.\s*', '', line)
                        html_lines.append(f"<li>{cleaned}</li>")
                html_lines.append("</ol>")
                html_paragraphs.append("\n".join(html_lines))
            elif all(line.strip().startswith("- ") for line in lines if line.strip()):
                # 无序列表
                html_lines = ["<ul>"]
                for line in lines:
                    line = line.strip()
                    if line.startswith("- "):
                        html_lines.append(f"<li>{line[2:]}</li>")
                html_lines.append("</ul>")
                html_paragraphs.append("\n".join(html_lines))
            else:
                # 普通段落，保留换行
                para_html = para.replace("\n", "<br>")
                html_paragraphs.append(f"<p>{para_html}</p>")

        return "\n".join(html_paragraphs)
