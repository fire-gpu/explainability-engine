"""
核心数据结构模块

定义了整个可解释性分析引擎中使用的所有核心数据类（dataclass），
涵盖数据元信息、分析配置、描述性统计、因果分析、预测模拟及最终报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


# ============================================================
# 数据元信息
# ============================================================


@dataclass
class ColumnMeta:
    """列元信息

    Attributes:
        name: 列名
        type: 列的数据类型，支持 numeric / categorical / datetime / text
        description: 列的业务含义描述
    """

    name: str
    type: Literal["numeric", "categorical", "datetime", "text"]
    description: str = ""


@dataclass
class DataMetadata:
    """数据集元信息

    Attributes:
        columns: 各列的元信息列表
        row_count: 数据行数
        time_range: 时间范围（可选），格式如 ("2020-01-01", "2023-12-31")
        missing_ratio: 整体缺失比例，取值 [0, 1]
        domain: 业务领域描述，如 "金融"、"电商" 等
    """

    columns: list[ColumnMeta] = field(default_factory=list)
    row_count: int = 0
    time_range: tuple[str, str] | None = None
    missing_ratio: float = 0.0
    domain: str = ""


# ============================================================
# 分析配置
# ============================================================


@dataclass
class AnalysisConfig:
    """分析配置

    Attributes:
        audience: 目标受众，如 "executive" / "analyst" / "technical"
        depth: 分析深度，如 "quick" / "standard" / "deep"
        target_variable: 目标变量名（可选）
        causal_enabled: 是否启用因果分析
        predictive_enabled: 是否启用预测模拟
        business_question: 用户输入的关键业务问题
        business_context: 用户描述的业务场景上下文
    """

    audience: str = "analyst"
    depth: str = "standard"
    target_variable: str | None = None
    causal_enabled: bool = True
    predictive_enabled: bool = True
    business_question: str = ""
    business_context: str = ""


# ============================================================
# 业务理解（L1 数据预扫描结果）
# ============================================================


@dataclass
class BusinessUnderstanding:
    """LLM 对数据的业务理解结果

    通过 L1 数据预扫描阶段，让 LLM 先"阅读"数据并推断业务语义，
    将理解结果注入后续分析流程，提升解释的业务相关性。

    Attributes:
        inferred_scenario: LLM 推断的业务场景描述
        key_metrics: 识别出的关键业务指标列表
        causal_hypotheses: LLM 生成的因果假设列表
        data_characteristics: 数据特征描述
        analysis_suggestions: 分析建议
        business_question_answer: 对用户业务问题的初步回答
    """

    inferred_scenario: str = ""
    key_metrics: list[str] = field(default_factory=list)
    causal_hypotheses: list[str] = field(default_factory=list)
    data_characteristics: str = ""
    analysis_suggestions: list[str] = field(default_factory=list)
    business_question_answer: str = ""


# ============================================================
# 分析输入
# ============================================================


@dataclass
class AnalysisInput:
    """分析输入，封装数据、元信息和配置

    Attributes:
        data: 原始数据 DataFrame
        metadata: 数据集元信息
        config: 分析配置
        business_understanding: L1 数据预扫描的业务理解结果（可选）
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: DataMetadata = field(default_factory=DataMetadata)
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    business_understanding: BusinessUnderstanding | None = None


# ============================================================
# 描述性分析结果
# ============================================================


@dataclass
class VarImportance:
    """变量重要性

    Attributes:
        name: 变量名
        score: 重要性得分
        rank: 重要性排名
    """

    name: str
    score: float
    rank: int


@dataclass
class Distribution:
    """单变量分布统计

    Attributes:
        column: 列名
        mean: 均值
        median: 中位数
        std: 标准差
        skewness: 偏度
        kurtosis: 峰度
        min: 最小值
        max: 最大值
        outlier_count: 异常值数量
        outlier_ratio: 异常值比例
    """

    column: str
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    min: float | None = None
    max: float | None = None
    outlier_count: int = 0
    outlier_ratio: float = 0.0


@dataclass
class CorrelationPair:
    """相关性对

    Attributes:
        var1: 变量1
        var2: 变量2
        coefficient: 相关系数
        method: 计算方法，如 "pearson" / "spearman"
    """

    var1: str
    var2: str
    coefficient: float
    method: str = "pearson"


@dataclass
class Anomaly:
    """异常记录

    Attributes:
        column: 所在列
        value: 异常值
        index: 行索引
        type: 异常类型，如 "outlier" / "missing" / "invalid"
    """

    column: str
    value: Any
    index: int
    type: str = "outlier"


@dataclass
class DescriptionResult:
    """描述性分析结果

    Attributes:
        variable_importance: 变量重要性列表
        distributions: 各变量分布统计
        correlations: 相关性对列表
        anomalies: 异常记录列表
        narrative: 自然语言叙述
    """

    variable_importance: list[VarImportance] = field(default_factory=list)
    distributions: list[Distribution] = field(default_factory=list)
    correlations: list[CorrelationPair] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    narrative: str = ""


# ============================================================
# 因果分析结果
# ============================================================


@dataclass
class CausalEdge:
    """因果边

    Attributes:
        from_node: 原因节点
        to_node: 结果节点
        effect_size: 效应大小
        p_value: 统计显著性 p 值
        method: 识别方法
    """

    from_node: str
    to_node: str
    effect_size: float
    p_value: float
    method: str = ""


@dataclass
class CausalGraph:
    """因果图

    Attributes:
        nodes: 节点列表
        edges: 因果边列表
        method: 因果发现方法
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[CausalEdge] = field(default_factory=list)
    method: str = ""


@dataclass
class CausalEffect:
    """因果效应估计

    Attributes:
        treatment: 处理变量
        outcome: 结果变量
        effect_size: 平均处理效应 (ATE)
        confidence_interval: 置信区间，如 (lower, upper)
        p_value: p 值
        method: 估计方法
    """

    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    method: str = ""


@dataclass
class Counterfactual:
    """反事实推理结果

    Attributes:
        treatment: 处理变量
        original_value: 原始值
        counterfactual_value: 反事实值
        predicted_outcome: 反事实预测结果
        original_outcome: 原始结果
    """

    treatment: str
    original_value: Any
    counterfactual_value: Any
    predicted_outcome: float | None = None
    original_outcome: float | None = None


@dataclass
class CausalResult:
    """因果分析结果

    Attributes:
        causal_graph: 因果图
        causal_effects: 因果效应列表
        counterfactuals: 反事实推理列表
        narrative: 自然语言叙述
    """

    causal_graph: CausalGraph | None = None
    causal_effects: list[CausalEffect] = field(default_factory=list)
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    narrative: str = ""


# ============================================================
# 预测模拟结果
# ============================================================


@dataclass
class Scenario:
    """模拟场景

    Attributes:
        name: 场景名称
        description: 场景描述
        parameters: 场景参数（变量名 -> 值）
        predicted_outcomes: 预测结果（变量名 -> 值）
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    predicted_outcomes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SensitivityEntry:
    """敏感性分析条目

    Attributes:
        variable: 变量名
        sensitivity_score: 敏感性得分
        direction: 影响方向，如 "positive" / "negative" / "neutral"
        impact_range: 影响范围
    """

    variable: str
    sensitivity_score: float
    direction: str = "neutral"
    impact_range: tuple[float, float] | None = None


@dataclass
class SensitivityAnalysis:
    """敏感性分析结果

    Attributes:
        entries: 敏感性条目列表
        method: 分析方法
    """

    entries: list[SensitivityEntry] = field(default_factory=list)
    method: str = ""


@dataclass
class WhatIf:
    """What-If 分析结果

    Attributes:
        variable: 变量名
        original_value: 原始值
        new_value: 新值
        outcome_change: 结果变化量
        confidence: 置信度
    """

    variable: str
    original_value: Any
    new_value: Any
    outcome_change: float | None = None
    confidence: float | None = None


@dataclass
class PredictionResult:
    """预测模拟结果

    Attributes:
        scenarios: 模拟场景列表
        sensitivity: 敏感性分析结果
        what_ifs: What-If 分析列表
        narrative: 自然语言叙述
    """

    scenarios: list[Scenario] = field(default_factory=list)
    sensitivity: SensitivityAnalysis | None = None
    what_ifs: list[WhatIf] = field(default_factory=list)
    narrative: str = ""


# ============================================================
# 图表
# ============================================================


@dataclass
class Chart:
    """图表定义

    Attributes:
        type: 图表类型，如 "bar" / "line" / "scatter" / "heatmap"
        title: 图表标题
        data: 图表数据（结构因类型而异）
        description: 图表描述
    """

    type: str
    title: str
    data: dict[str, Any] = field(default_factory=dict)
    description: str = ""


# ============================================================
# 报告
# ============================================================


@dataclass
class ReportMetadata:
    """报告元信息

    Attributes:
        title: 报告标题
        generated_at: 生成时间 (ISO 格式字符串)
        domain: 业务领域
        data_source: 数据来源描述
        analysis_config: 分析配置快照
    """

    title: str = ""
    generated_at: str = ""
    domain: str = ""
    data_source: str = ""
    analysis_config: AnalysisConfig | None = None


@dataclass
class ExplainabilityReport:
    """可解释性分析报告（最终输出）

    Attributes:
        executive_summary: 高管摘要（面向非技术受众）
        detailed_analysis: 详细分析（面向分析师）
        technical_appendix: 技术附录（面向技术人员）
        charts: 图表列表
        metadata: 报告元信息
    """

    executive_summary: str = ""
    detailed_analysis: str = ""
    technical_appendix: str = ""
    charts: list[Chart] = field(default_factory=list)
    metadata: ReportMetadata | None = None


# ============================================================
# 状态日志
# ============================================================


@dataclass
class StateLog:
    """状态转换日志

    Attributes:
        state: 当前状态
        timestamp: 时间戳 (ISO 格式字符串)
        input_summary: 输入摘要
        output_summary: 输出摘要
        duration_ms: 该阶段耗时（毫秒）
        error: 错误信息（如有）
    """

    state: str
    timestamp: str = ""
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: float = 0.0
    error: str | None = None
