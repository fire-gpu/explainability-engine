"""
描述性分析器测试模块

测试变量重要性排名、分布统计、相关性计算、异常值检测和叙述生成（无 LLM 模式）。
"""

import pandas as pd
import pytest

from analyzers.descriptive import DescriptiveAnalyzer
from core.models import AnalysisConfig, AnalysisInput, ColumnMeta, DataMetadata


@pytest.fixture
def sample_data():
    """创建测试用数据集和分析输入"""
    df = pd.DataFrame({
        "revenue": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                     600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
        "cost": [80, 100, 130, 160, 190, 220, 250, 280, 310, 340,
                  370, 400, 430, 460, 490, 520, 550, 580, 610, 640],
        "profit": [20, 50, 70, 90, 110, 130, 150, 170, 190, 210,
                    230, 250, 270, 290, 310, 330, 350, 370, 390, 410],
        "region": ["华东", "华北", "华东", "华南", "华北",
                    "华东", "华南", "华北", "华东", "华南",
                    "华东", "华北", "华南", "华东", "华北",
                    "华东", "华南", "华北", "华东", "华南"],
    })

    columns = [
        ColumnMeta(name="revenue", type="numeric", description="收入"),
        ColumnMeta(name="cost", type="numeric", description="成本"),
        ColumnMeta(name="profit", type="numeric", description="利润"),
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
        target_variable="profit",
    )

    return AnalysisInput(data=df, metadata=metadata, config=config)


class TestDescriptiveAnalyzer:
    """描述性分析器测试类"""

    def test_variable_importance(self, sample_data):
        """测试变量重要性排名"""
        analyzer = DescriptiveAnalyzer()
        result = analyzer.analyze(sample_data)

        # 验证变量重要性列表不为空
        assert len(result.variable_importance) > 0

        # 验证排名从 1 开始且连续
        ranks = [item.rank for item in result.variable_importance]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

        # 验证得分非负
        for item in result.variable_importance:
            assert item.score >= 0

    def test_distributions(self, sample_data):
        """测试分布统计"""
        analyzer = DescriptiveAnalyzer()
        result = analyzer.analyze(sample_data)

        # 验证分布统计不为空
        assert len(result.distributions) > 0

        # 验证每个数值变量都有分布统计
        dist_columns = {d.column for d in result.distributions}
        assert "revenue" in dist_columns
        assert "cost" in dist_columns
        assert "profit" in dist_columns

        # 验证统计量合理
        for dist in result.distributions:
            assert dist.mean is not None
            assert dist.median is not None
            assert dist.std is not None
            assert dist.min is not None
            assert dist.max is not None
            assert dist.min <= dist.mean <= dist.max

    def test_correlations(self, sample_data):
        """测试相关性计算"""
        analyzer = DescriptiveAnalyzer()
        result = analyzer.analyze(sample_data)

        # 验证相关性列表不为空（revenue/cost/profit 之间应该有强相关性）
        assert len(result.correlations) > 0

        # 验证相关系数在 [-1, 1] 范围内
        for corr in result.correlations:
            assert -1.0 <= corr.coefficient <= 1.0
            assert corr.method in ("pearson", "spearman")

    def test_anomaly_detection(self, sample_data):
        """测试异常值检测"""
        analyzer = DescriptiveAnalyzer()
        result = analyzer.analyze(sample_data)

        # 验证异常值列表存在（可能为空，取决于数据分布）
        assert isinstance(result.anomalies, list)

        # 如果有异常值，验证其结构
        for anomaly in result.anomalies:
            assert anomaly.column in ["revenue", "cost", "profit"]
            assert anomaly.type == "outlier"
            assert isinstance(anomaly.index, int)

    def test_narrative_generation(self, sample_data):
        """测试叙述生成（无 LLM 模式）"""
        # 不传入 LLM 客户端，使用模板回退
        analyzer = DescriptiveAnalyzer(llm_client=None)
        result = analyzer.analyze(sample_data)

        # 验证叙述不为空
        assert result.narrative != ""
        assert len(result.narrative) > 10

        # 验证叙述包含关键信息
        assert "分析" in result.narrative or "变量" in result.narrative
