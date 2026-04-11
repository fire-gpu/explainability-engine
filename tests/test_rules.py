"""
规则引擎测试模块

测试规则的添加与评估、流程控制规则、校验规则和触发动作。
"""

import pytest

from rules.engine import Rule, RulesEngine


class TestRulesEngine:
    """规则引擎测试类"""

    def test_add_and_evaluate(self):
        """测试添加规则并评估"""
        engine = RulesEngine()

        # 添加一条简单规则
        rule = Rule(
            name="变量数过多",
            rule_type="flow",
            condition=lambda ctx: ctx.get("variable_count", 0) > 50,
            action=lambda ctx: {"skip_causal_graph": True},
            description="变量数超过 50 时跳过因果图可视化",
        )
        engine.add_rule(rule)

        # 评估：变量数 = 60，应触发
        results = engine.evaluate({"variable_count": 60})
        assert len(results) == 1
        assert results[0][1] is True  # 规则被触发

        # 评估：变量数 = 30，不应触发
        results = engine.evaluate({"variable_count": 30})
        assert len(results) == 1
        assert results[0][1] is False  # 规则未触发

    def test_flow_rules(self):
        """测试流程控制规则"""
        engine = RulesEngine()

        # 添加流程控制规则
        engine.add_rule(Rule(
            name="样本量不足跳过因果",
            rule_type="flow",
            condition=lambda ctx: ctx.get("row_count", 0) < 30,
            action=lambda ctx: {
                "target": "causal_inference",
                "reason": "样本量不足",
            },
            description="样本量不足 30 时跳过因果分析",
        ))

        engine.add_rule(Rule(
            name="缺失率过高跳过预测",
            rule_type="flow",
            condition=lambda ctx: ctx.get("missing_ratio", 0) > 0.3,
            action=lambda ctx: {
                "target": "predictive_simulation",
                "reason": "缺失率过高",
            },
            description="缺失率超过 30% 时跳过预测模拟",
        ))

        # 获取流程控制规则
        flow_rules = engine.get_rules_by_type("flow")
        assert len(flow_rules) == 2

        # 评估样本量不足的场景
        context = {"row_count": 20, "missing_ratio": 0.1}
        actions = engine.get_triggered_actions(context)
        assert len(actions) == 1
        assert actions[0][0] == "样本量不足跳过因果"

    def test_validation_rules(self):
        """测试校验规则"""
        engine = RulesEngine()

        # 添加校验规则
        engine.add_rule(Rule(
            name="目标变量存在性校验",
            rule_type="validation",
            condition=lambda ctx: ctx.get("target_variable") is not None,
            action=lambda ctx: {"valid": True},
            description="校验目标变量是否已设置",
        ))

        engine.add_rule(Rule(
            name="数据量校验",
            rule_type="validation",
            condition=lambda ctx: ctx.get("row_count", 0) >= 10,
            action=lambda ctx: {"valid": True, "message": "数据量充足"},
            description="校验数据量是否足够",
        ))

        # 获取校验规则
        validation_rules = engine.get_rules_by_type("validation")
        assert len(validation_rules) == 2

        # 评估：两个条件都满足
        context = {"target_variable": "revenue", "row_count": 100}
        results = engine.evaluate(context)
        assert all(triggered for _, triggered in results)

        # 评估：目标变量未设置
        context2 = {"target_variable": None, "row_count": 100}
        results2 = engine.evaluate(context2)
        assert results2[0][1] is False  # 目标变量校验失败
        assert results2[1][1] is True   # 数据量校验通过

    def test_triggered_actions(self):
        """测试触发动作"""
        engine = RulesEngine()

        # 添加多条规则
        engine.add_rule(Rule(
            name="高缺失率警告",
            rule_type="validation",
            condition=lambda ctx: ctx.get("missing_ratio", 0) > 0.2,
            action=lambda ctx: {"level": "warning", "message": "缺失率较高"},
        ))

        engine.add_rule(Rule(
            name="变量数警告",
            rule_type="validation",
            condition=lambda ctx: ctx.get("variable_count", 0) > 100,
            action=lambda ctx: {"level": "warning", "message": "变量数过多"},
        ))

        # 评估：只有高缺失率触发
        context = {"missing_ratio": 0.5, "variable_count": 50}
        actions = engine.get_triggered_actions(context)
        assert len(actions) == 1
        assert actions[0][0] == "高缺失率警告"
        assert actions[0][1] == {"level": "warning", "message": "缺失率较高"}

        # 评估：两条都触发
        context2 = {"missing_ratio": 0.5, "variable_count": 150}
        actions2 = engine.get_triggered_actions(context2)
        assert len(actions2) == 2

        # 评估：都不触发
        context3 = {"missing_ratio": 0.1, "variable_count": 50}
        actions3 = engine.get_triggered_actions(context3)
        assert len(actions3) == 0

    def test_remove_rule(self):
        """测试移除规则"""
        engine = RulesEngine()
        engine.add_rule(Rule(
            name="测试规则",
            rule_type="flow",
            condition=lambda ctx: True,
            action=lambda ctx: None,
        ))

        assert len(engine.rules) == 1
        removed = engine.remove_rule("测试规则")
        assert removed is True
        assert len(engine.rules) == 0

        # 移除不存在的规则
        removed = engine.remove_rule("不存在的规则")
        assert removed is False

    def test_clear(self):
        """测试清空所有规则"""
        engine = RulesEngine()
        engine.add_rule(Rule(
            name="规则1",
            rule_type="flow",
            condition=lambda ctx: True,
            action=lambda ctx: None,
        ))
        engine.add_rule(Rule(
            name="规则2",
            rule_type="validation",
            condition=lambda ctx: True,
            action=lambda ctx: None,
        ))

        assert len(engine.rules) == 2
        engine.clear()
        assert len(engine.rules) == 0
