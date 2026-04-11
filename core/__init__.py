"""
核心模块

导出核心数据结构、状态机和编排器。
"""

from core.models import (
    AnalysisConfig,
    AnalysisInput,
    Anomaly,
    CausalEdge,
    CausalEffect,
    CausalGraph,
    CausalResult,
    Chart,
    ColumnMeta,
    CorrelationPair,
    Counterfactual,
    DataMetadata,
    DescriptionResult,
    Distribution,
    ExplainabilityReport,
    PredictionResult,
    ReportMetadata,
    Scenario,
    SensitivityAnalysis,
    SensitivityEntry,
    StateLog,
    VarImportance,
    WhatIf,
)
from core.orchestrator import Orchestrator
from core.state_machine import AnalysisState, StateMachine

__all__ = [
    # 数据元信息
    "ColumnMeta",
    "DataMetadata",
    # 分析配置与输入
    "AnalysisConfig",
    "AnalysisInput",
    # 描述性分析
    "VarImportance",
    "Distribution",
    "CorrelationPair",
    "Anomaly",
    "DescriptionResult",
    # 因果分析
    "CausalEdge",
    "CausalGraph",
    "CausalEffect",
    "Counterfactual",
    "CausalResult",
    # 预测模拟
    "Scenario",
    "SensitivityEntry",
    "SensitivityAnalysis",
    "WhatIf",
    "PredictionResult",
    # 图表与报告
    "Chart",
    "ReportMetadata",
    "ExplainabilityReport",
    # 状态
    "StateLog",
    "AnalysisState",
    "StateMachine",
    # 编排器
    "Orchestrator",
]
