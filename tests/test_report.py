"""
报告生成器测试模块

测试报告生成、Markdown 渲染、HTML 渲染和执行摘要长度限制。
"""

import pandas as pd
import pytest

from analyzers.descriptive import DescriptiveAnalyzer
from core.models import (
    AnalysisConfig,
    AnalysisInput,
    ColumnMeta,
    DataMetadata,
    DescriptionResult,
    Distribution,
    VarImportance,
)
from report.generator import ReportGenerator
from report.renderers.html import HTMLRenderer
from report.renderers.markdown import MarkdownRenderer


@pytest.fixture
def sample_analysis_input():
    """创建测试用分析输入"""
    df = pd.DataFrame({
        "sales": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "marketing_spend": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        "region": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })

    columns = [
        ColumnMeta(name="sales", type="numeric", description="销售额"),
        ColumnMeta(name="marketing_spend", type="numeric", description="营销支出"),
        ColumnMeta(name="region", type="categorical", description="区域"),
    ]

    metadata = DataMetadata(
        columns=columns,
        row_count=len(df),
        missing_ratio=0.0,
        domain="通用",
    )

    config = AnalysisConfig(
        audience="analyst",
        depth="standard",
        target_variable="sales",
    )

    return AnalysisInput(data=df, metadata=metadata, config=config)


@pytest.fixture
def sample_desc_result():
    """创建测试用描述性分析结果"""
    return DescriptionResult(
        variable_importance=[
            VarImportance(name="marketing_spend", score=0.65, rank=1),
            VarImportance(name="region", score=0.35, rank=2),
        ],
        distributions=[
            Distribution(
                column="sales",
                mean=550.0,
                median=550.0,
                std=302.765,
                skewness=0.0,
                kurtosis=-1.2,
                min=100.0,
                max=1000.0,
                outlier_count=0,
                outlier_ratio=0.0,
            ),
        ],
        correlations=[],
        anomalies=[],
        narrative="测试叙述",
    )


class TestReportGenerator:
    """报告生成器测试类"""

    def test_report_generation(self, sample_analysis_input, sample_desc_result):
        """测试报告生成"""
        generator = ReportGenerator(llm_client=None)
        report = generator.generate(
            analysis_input=sample_analysis_input,
            desc_result=sample_desc_result,
            causal_result=None,
            pred_result=None,
        )

        # 验证报告结构完整
        assert report.executive_summary != ""
        assert report.detailed_analysis != ""
        assert report.technical_appendix != ""

        # 验证元信息
        assert report.metadata is not None
        assert report.metadata.title != ""
        assert report.metadata.generated_at != ""

    def test_markdown_render(self, sample_analysis_input, sample_desc_result):
        """测试 Markdown 渲染"""
        generator = ReportGenerator(llm_client=None)
        report = generator.generate(
            analysis_input=sample_analysis_input,
            desc_result=sample_desc_result,
            causal_result=None,
            pred_result=None,
        )

        renderer = MarkdownRenderer()
        markdown_text = renderer.render(report)

        # 验证 Markdown 格式
        assert "# " in markdown_text  # 标题
        assert "## " in markdown_text  # 章节标题
        assert "执行摘要" in markdown_text
        assert "详细分析" in markdown_text
        assert "技术附录" in markdown_text

    def test_html_render(self, sample_analysis_input, sample_desc_result):
        """测试 HTML 渲染"""
        generator = ReportGenerator(llm_client=None)
        report = generator.generate(
            analysis_input=sample_analysis_input,
            desc_result=sample_desc_result,
            causal_result=None,
            pred_result=None,
        )

        renderer = HTMLRenderer()
        html_text = renderer.render(report)

        # 验证 HTML 格式
        assert "<!DOCTYPE html>" in html_text
        assert "<html" in html_text
        assert "</html>" in html_text
        assert "<h1>" in html_text
        assert "<style>" in html_text
        assert "执行摘要" in html_text

    def test_executive_summary_length(self, sample_analysis_input, sample_desc_result):
        """测试执行摘要长度限制（无 LLM 模式下模板生成的摘要也应合理）"""
        generator = ReportGenerator(llm_client=None)
        report = generator.generate(
            analysis_input=sample_analysis_input,
            desc_result=sample_desc_result,
            causal_result=None,
            pred_result=None,
        )

        # 执行摘要不应为空
        assert len(report.executive_summary) > 0

        # 无 LLM 模式下模板生成的摘要长度应合理（不超过 500 字）
        assert len(report.executive_summary) < 500
