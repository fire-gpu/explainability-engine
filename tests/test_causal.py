"""
因果推断引擎测试模块

测试 CausalAnalyzer 的核心功能，包括：
- 初始化（有/无 DoWhy）
- 因果图转 DOT 格式
- 无 DoWhy 时的 OLS 回退分析
- 有 DoWhy 时的因果分析（如果可用）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models import (
    AnalysisConfig,
    AnalysisInput,
    CausalEdge,
    CausalEffect,
    CausalGraph,
    CorrelationPair,
    DataMetadata,
    DescriptionResult,
)


# ============================================================
# 测试数据工厂
# ============================================================


def _make_test_df(n: int = 200) -> pd.DataFrame:
    """创建测试用数据框

    生成包含多个数值变量的模拟数据，变量之间存在已知的因果关系：
    - budget（预算）影响 clicks（点击量）
    - clicks 影响 conversions（转化量）
    - season（季节）同时影响 budget 和 conversions（混杂因子）

    Args:
        n: 样本数量

    Returns:
        pd.DataFrame: 测试数据框
    """
    np.random.seed(42)
    season = np.random.choice([1, 2, 3, 4], size=n)
    budget = 1000 + season * 200 + np.random.normal(0, 100, n)
    clicks = 50 + budget * 0.05 + np.random.normal(0, 10, n)
    conversions = 5 + clicks * 0.1 + season * 2 + np.random.normal(0, 2, n)

    return pd.DataFrame({
        "season": season,
        "budget": budget,
        "clicks": clicks,
        "conversions": conversions,
    })


def _make_test_correlations() -> list[CorrelationPair]:
    """创建测试用相关性对列表

    Returns:
        list[CorrelationPair]: 模拟的相关性对
    """
    return [
        CorrelationPair(var1="budget", var2="clicks", coefficient=0.85, method="pearson"),
        CorrelationPair(var1="clicks", var2="conversions", coefficient=0.78, method="pearson"),
        CorrelationPair(var1="season", var2="budget", coefficient=0.65, method="pearson"),
        CorrelationPair(var1="season", var2="conversions", coefficient=0.55, method="pearson"),
    ]


def _make_analysis_input(
    df: pd.DataFrame | None = None,
    target: str = "conversions",
) -> AnalysisInput:
    """创建测试用分析输入

    Args:
        df: 数据框，为 None 时自动生成
        target: 目标变量名

    Returns:
        AnalysisInput: 测试分析输入
    """
    if df is None:
        df = _make_test_df()

    metadata = DataMetadata(
        columns=[],
        row_count=len(df),
        domain="电商广告",
    )

    config = AnalysisConfig(
        target_variable=target,
        audience="analyst",
        causal_enabled=True,
    )

    return AnalysisInput(
        data=df,
        metadata=metadata,
        config=config,
    )


def _make_desc_result() -> DescriptionResult:
    """创建测试用描述性分析结果

    Returns:
        DescriptionResult: 模拟的描述性分析结果
    """
    return DescriptionResult(
        correlations=_make_test_correlations(),
    )


# ================================================================
# 测试用例
# ================================================================


class TestCausalAnalyzerInit:
    """CausalAnalyzer 初始化测试"""

    def test_init_with_no_llm(self):
        """测试无 LLM 客户端的初始化"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer(llm_client=None)
        assert analyzer.llm is None
        assert isinstance(analyzer._dowhy_available, bool)

    def test_init_dowhy_available(self):
        """测试 DoWhy 可用性检测"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        # 在当前环境中 dowhy 已安装
        try:
            import dowhy  # noqa: F401
            assert analyzer._dowhy_available is True
        except ImportError:
            assert analyzer._dowhy_available is False


class TestGraphToDot:
    """因果图转 DOT 格式测试"""

    def test_empty_graph(self):
        """测试空因果图的 DOT 转换"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        graph = CausalGraph(nodes=[], edges=[])
        dot = analyzer._graph_to_dot(graph)
        assert dot == "digraph {}"

    def test_single_edge(self):
        """测试单条边的因果图 DOT 转换"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        graph = CausalGraph(
            nodes=["A", "B"],
            edges=[CausalEdge(from_node="A", to_node="B", effect_size=0.5, p_value=0.01)],
        )
        dot = analyzer._graph_to_dot(graph)
        assert "A -> B;" in dot
        assert "digraph" in dot

    def test_multiple_edges(self):
        """测试多条边的因果图 DOT 转换"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        graph = CausalGraph(
            nodes=["cost", "price", "demand"],
            edges=[
                CausalEdge(from_node="cost", to_node="price", effect_size=0.8, p_value=0.001),
                CausalEdge(from_node="price", to_node="demand", effect_size=-0.6, p_value=0.005),
            ],
        )
        dot = analyzer._graph_to_dot(graph)
        assert "cost -> price;" in dot
        assert "price -> demand;" in dot
        assert "digraph" in dot

    def test_dot_format_valid(self):
        """测试 DOT 格式的基本结构"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        graph = CausalGraph(
            nodes=["X", "Y", "Z"],
            edges=[
                CausalEdge(from_node="X", to_node="Y", effect_size=0.5, p_value=0.01),
                CausalEdge(from_node="Y", to_node="Z", effect_size=0.3, p_value=0.05),
            ],
        )
        dot = analyzer._graph_to_dot(graph)
        # 验证 DOT 格式的基本结构
        assert dot.startswith("digraph {")
        assert dot.endswith("}")
        assert dot.count("->") == 2
        assert dot.count(";") == 2


class TestCausalAnalysisWithoutDoWhy:
    """无 DoWhy 时的回退测试（通过模拟 _dowhy_available=False）"""

    def test_fallback_analysis(self):
        """测试 OLS 回退方案的完整分析流程"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        # 强制使用 OLS 回退
        analyzer._dowhy_available = False

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        # 验证结果结构
        assert result is not None
        assert result.causal_graph is not None
        assert isinstance(result.causal_effects, list)
        assert isinstance(result.counterfactuals, list)
        assert isinstance(result.narrative, str)

    def test_fallback_effects_method(self):
        """测试回退方案的效应方法标记为 ols"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        analyzer._dowhy_available = False

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        # 所有效应的方法应为 "ols"
        for eff in result.causal_effects:
            assert eff.method == "ols"
            assert eff.identified is False
            assert eff.refutation_results is None

    def test_fallback_effects_sorted(self):
        """测试回退方案的效应按大小排序"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        analyzer._dowhy_available = False

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        if len(result.causal_effects) >= 2:
            sizes = [abs(e.effect_size) for e in result.causal_effects]
            assert sizes == sorted(sizes, reverse=True)

    def test_fallback_empty_correlations(self):
        """测试无相关性时的回退"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        analyzer._dowhy_available = False

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = DescriptionResult(correlations=[])

        result = analyzer.analyze(analysis_input, desc_result)

        assert result.causal_graph is not None
        assert len(result.causal_graph.edges) == 0
        assert len(result.causal_effects) == 0

    def test_fallback_narrative_not_empty(self):
        """测试回退方案生成非空叙述"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        analyzer._dowhy_available = False

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        assert len(result.narrative) > 0
        # 回退叙述应包含"OLS"关键词
        assert "OLS" in result.narrative


class TestCausalAnalysisWithDoWhy:
    """有 DoWhy 时的因果分析测试"""

    def test_dowhy_analysis_runs(self):
        """测试 DoWhy 分析流程能正常运行"""
        try:
            import dowhy  # noqa: F401
        except ImportError:
            pytest.skip("DoWhy 未安装，跳过此测试")

        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        assert analyzer._dowhy_available is True

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        # 验证结果结构
        assert result is not None
        assert result.causal_graph is not None
        assert isinstance(result.causal_effects, list)
        assert isinstance(result.counterfactuals, list)
        assert isinstance(result.narrative, str)

    def test_dowhy_effects_have_method(self):
        """测试 DoWhy 效应的方法标记"""
        try:
            import dowhy  # noqa: F401
        except ImportError:
            pytest.skip("DoWhy 未安装，跳过此测试")

        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        # 至少应有一些效应
        assert len(result.causal_effects) > 0

        # DoWhy 效应应标记为 dowhy.backdoor.linear_regression
        dowhy_effects = [e for e in result.causal_effects if "dowhy" in (e.method or "")]
        assert len(dowhy_effects) > 0

        for eff in dowhy_effects:
            assert "dowhy" in eff.method
            assert eff.identified is True

    def test_dowhy_refutation_results(self):
        """测试 DoWhy 反驳测试结果"""
        try:
            import dowhy  # noqa: F401
        except ImportError:
            pytest.skip("DoWhy 未安装，跳过此测试")

        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        # 检查是否有反驳测试结果
        dowhy_effects = [e for e in result.causal_effects if e.refutation_results is not None]
        # 反驳测试是可选的，不一定所有效应都有
        for eff in dowhy_effects:
            assert isinstance(eff.refutation_results, dict)
            # 至少应有 placebo 测试
            assert "placebo" in eff.refutation_results

    def test_dowhy_narrative_mentions_dowhy(self):
        """测试 DoWhy 叙述中包含 DoWhy 相关描述"""
        try:
            import dowhy  # noqa: F401
        except ImportError:
            pytest.skip("DoWhy 未安装，跳过此测试")

        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()

        df = _make_test_df()
        analysis_input = _make_analysis_input(df)
        desc_result = _make_desc_result()

        result = analyzer.analyze(analysis_input, desc_result)

        # 叙述应包含 DoWhy 相关关键词
        assert "DoWhy" in result.narrative


class TestCausalEffectModel:
    """CausalEffect 数据结构测试"""

    def test_new_fields_exist(self):
        """测试新增字段存在且有默认值"""
        eff = CausalEffect(
            treatment="X",
            outcome="Y",
            effect_size=0.5,
        )
        assert eff.method == ""
        assert eff.refutation_results is None
        assert eff.identified is True

    def test_dowhy_effect_fields(self):
        """测试 DoWhy 效应的字段赋值"""
        eff = CausalEffect(
            treatment="budget",
            outcome="conversions",
            effect_size=0.1234,
            confidence_interval=(0.05, 0.20),
            p_value=0.001,
            method="dowhy.backdoor.linear_regression",
            refutation_results={
                "placebo": {"passed": True, "details": "安慰剂效应=0.0012"},
                "random_cause": {"passed": True, "details": "相对变化=3.5%"},
            },
            identified=True,
        )
        assert eff.method == "dowhy.backdoor.linear_regression"
        assert eff.refutation_results is not None
        assert eff.refutation_results["placebo"]["passed"] is True
        assert eff.identified is True

    def test_ols_effect_fields(self):
        """测试 OLS 效应的字段赋值"""
        eff = CausalEffect(
            treatment="X",
            outcome="Y",
            effect_size=0.5,
            method="ols",
            refutation_results=None,
            identified=False,
        )
        assert eff.method == "ols"
        assert eff.refutation_results is None
        assert eff.identified is False


class TestBuildCausalHypotheses:
    """因果假设图构建测试"""

    def test_with_domain_config(self):
        """测试带领域配置的因果假设图构建"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        analyzer._dowhy_available = False

        df = _make_test_df()
        correlations = _make_test_correlations()

        # 模拟带 domain_config 的 metadata
        class MockCausalTemplate:
            def __init__(self, cause: str, effect: str):
                self.cause = cause
                self.effect = effect

        class MockDomainConfig:
            def __init__(self):
                self.causal_templates = [
                    MockCausalTemplate(cause="brand_awareness", effect="conversions"),
                ]

        class MockMetadata:
            def __init__(self):
                self.domain_config = MockDomainConfig()

        metadata = MockMetadata()
        graph = analyzer._build_causal_hypotheses(df, correlations, "电商广告", metadata)

        # 验证领域模板的边被纳入图中
        edge_pairs = [(e.from_node, e.to_node) for e in graph.edges]
        assert ("brand_awareness", "conversions") in edge_pairs

    def test_without_domain_config(self):
        """测试无领域配置的因果假设图构建"""
        from analyzers.causal import CausalAnalyzer

        analyzer = CausalAnalyzer()
        analyzer._dowhy_available = False

        df = _make_test_df()
        correlations = _make_test_correlations()

        graph = analyzer._build_causal_hypotheses(df, correlations, "电商广告")

        # 验证图不为空
        assert len(graph.edges) > 0
        assert len(graph.nodes) > 0
