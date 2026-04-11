"""
校验规则模块

预定义分析结果的校验规则，确保输出质量满足阈值要求，
如因果效应置信度、变量重要性得分等。
"""

from __future__ import annotations

from typing import Any

from rules.engine import Rule, RulesEngine


def create_validation_rules() -> list[Rule]:
    """创建预定义的校验规则列表

    包含以下规则：
    - 因果效应置信区间 > 0.95 才纳入报告
    - 变量重要性 score > 0.01 才展示
    - 相关性系数绝对值 > 0.3 才纳入报告
    - 异常值比例 > 30% 时标记数据质量问题
    - 预测置信度 < 0.5 时添加不确定性警告

    Returns:
        list[Rule]: 校验规则列表
    """
    rules: list[Rule] = []

    # 因果效应置信区间 > 0.95 才纳入报告
    rules.append(
        Rule(
            name="causal_confidence_threshold",
            rule_type="validation",
            condition=lambda ctx: any(
                effect.get("confidence", 0) < 0.95
                for effect in ctx.get("causal_effects", [])
            ),
            action=lambda ctx: {
                "action": "filter",
                "target": "causal_effects",
                "filter_fn": "remove_low_confidence",
                "threshold": 0.95,
                "reason": "仅保留置信度 >= 0.95 的因果效应纳入报告",
            },
            description="因果效应置信度 >= 0.95 才纳入报告",
        )
    )

    # 变量重要性 score > 0.01 才展示
    rules.append(
        Rule(
            name="variable_importance_threshold",
            rule_type="validation",
            condition=lambda ctx: any(
                var.get("score", 0) < 0.01
                for var in ctx.get("variable_importance", [])
            ),
            action=lambda ctx: {
                "action": "filter",
                "target": "variable_importance",
                "filter_fn": "remove_low_score",
                "threshold": 0.01,
                "reason": "仅保留重要性得分 > 0.01 的变量展示",
            },
            description="变量重要性 score > 0.01 才展示",
        )
    )

    # 相关性系数绝对值 > 0.3 才纳入报告
    rules.append(
        Rule(
            name="correlation_threshold",
            rule_type="validation",
            condition=lambda ctx: any(
                abs(corr.get("coefficient", 0)) < 0.3
                for corr in ctx.get("correlations", [])
            ),
            action=lambda ctx: {
                "action": "filter",
                "target": "correlations",
                "filter_fn": "remove_weak_correlations",
                "threshold": 0.3,
                "reason": "仅保留相关系数绝对值 > 0.3 的相关性对纳入报告",
            },
            description="相关性系数绝对值 > 0.3 才纳入报告",
        )
    )

    # 异常值比例 > 30% 时标记数据质量问题
    rules.append(
        Rule(
            name="outlier_ratio_warning",
            rule_type="validation",
            condition=lambda ctx: ctx.get("outlier_ratio", 0) > 0.3,
            action=lambda ctx: {
                "action": "warn",
                "target": "data_quality",
                "message": f"异常值比例 ({ctx.get('outlier_ratio', 0):.1%}) 超过 30%，数据质量可能存在问题",
            },
            description="异常值比例 > 30% 时标记数据质量问题",
        )
    )

    # 预测置信度 < 0.5 时添加不确定性警告
    rules.append(
        Rule(
            name="prediction_confidence_warning",
            rule_type="validation",
            condition=lambda ctx: any(
                scenario.get("confidence", 1.0) < 0.5
                for scenario in ctx.get("scenarios", [])
            ),
            action=lambda ctx: {
                "action": "warn",
                "target": "prediction_uncertainty",
                "message": "部分预测场景的置信度低于 0.5，结果不确定性较高",
            },
            description="预测置信度 < 0.5 时添加不确定性警告",
        )
    )

    return rules


def register_validation_rules(engine: RulesEngine) -> None:
    """将校验规则注册到规则引擎

    Args:
        engine: 目标规则引擎实例
    """
    for rule in create_validation_rules():
        engine.add_rule(rule)
