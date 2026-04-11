"""
流程控制规则模块

预定义分析流程中的控制规则，根据数据特征自动调整分析策略，
如跳过计算密集型步骤或降级分析深度。
"""

from __future__ import annotations

from typing import Any

from rules.engine import Rule, RulesEngine


def create_flow_rules() -> list[Rule]:
    """创建预定义的流程控制规则列表

    包含以下规则：
    - 变量数 > 50 时跳过因果图可视化
    - 缺失值 > 50% 时降级分析深度
    - 行数 < 30 时跳过因果推断
    - 变量数 > 100 时跳过相关性矩阵计算
    - 行数 < 10 时仅生成基础统计

    Returns:
        list[Rule]: 流程控制规则列表
    """
    rules: list[Rule] = []

    # 变量数 > 50 时跳过因果图可视化
    rules.append(
        Rule(
            name="skip_causal_graph_viz",
            rule_type="flow",
            condition=lambda ctx: ctx.get("variable_count", 0) > 50,
            action=lambda ctx: {
                "action": "skip",
                "target": "causal_graph_visualization",
                "reason": f"变量数 ({ctx.get('variable_count', 0)}) 超过 50，跳过因果图可视化以避免计算复杂度过高",
            },
            description="变量数超过 50 时跳过因果图可视化",
        )
    )

    # 缺失值 > 50% 时降级分析深度
    rules.append(
        Rule(
            name="degrade_analysis_depth",
            rule_type="flow",
            condition=lambda ctx: ctx.get("missing_ratio", 0) > 0.5,
            action=lambda ctx: {
                "action": "downgrade",
                "target": "analysis_depth",
                "value": "quick",
                "reason": f"缺失值比例 ({ctx.get('missing_ratio', 0):.1%}) 超过 50%，降级为快速分析模式",
            },
            description="缺失值超过 50% 时降级分析深度",
        )
    )

    # 行数 < 30 时跳过因果推断
    rules.append(
        Rule(
            name="skip_causal_inference",
            rule_type="flow",
            condition=lambda ctx: ctx.get("row_count", 0) < 30,
            action=lambda ctx: {
                "action": "skip",
                "target": "causal_inference",
                "reason": f"数据行数 ({ctx.get('row_count', 0)}) 少于 30，样本量不足以进行可靠的因果推断",
            },
            description="行数少于 30 时跳过因果推断",
        )
    )

    # 变量数 > 100 时跳过相关性矩阵计算
    rules.append(
        Rule(
            name="skip_correlation_matrix",
            rule_type="flow",
            condition=lambda ctx: ctx.get("variable_count", 0) > 100,
            action=lambda ctx: {
                "action": "skip",
                "target": "correlation_matrix",
                "reason": f"变量数 ({ctx.get('variable_count', 0)}) 超过 100，跳过全量相关性矩阵计算",
            },
            description="变量数超过 100 时跳过相关性矩阵计算",
        )
    )

    # 行数 < 10 时仅生成基础统计
    rules.append(
        Rule(
            name="basic_stats_only",
            rule_type="flow",
            condition=lambda ctx: ctx.get("row_count", 0) < 10,
            action=lambda ctx: {
                "action": "downgrade",
                "target": "analysis_mode",
                "value": "basic_stats",
                "reason": f"数据行数 ({ctx.get('row_count', 0)}) 少于 10，仅生成基础统计描述",
            },
            description="行数少于 10 时仅生成基础统计",
        )
    )

    return rules


def register_flow_rules(engine: RulesEngine) -> None:
    """将流程控制规则注册到规则引擎

    Args:
        engine: 目标规则引擎实例
    """
    for rule in create_flow_rules():
        engine.add_rule(rule)
