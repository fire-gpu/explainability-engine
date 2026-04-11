"""
状态机测试模块

测试分析引擎状态机的初始化、正常流程、跳过逻辑、错误处理和历史记录。
"""

import pytest

from core.state_machine import AnalysisState, StateMachine


class TestStateMachine:
    """状态机测试类"""

    def test_initial_state(self):
        """测试初始状态为 IDLE"""
        sm = StateMachine()
        assert sm.current_state == AnalysisState.IDLE

    def test_normal_flow(self):
        """测试正常流程: IDLE -> PARSING -> DESCRIPTIVE -> CAUSAL -> PREDICTIVE -> REPORTING -> DONE"""
        sm = StateMachine()

        sm.transition("start")
        assert sm.current_state == AnalysisState.PARSING

        sm.transition("parse_done")
        assert sm.current_state == AnalysisState.DESCRIPTIVE

        sm.transition("descriptive_done")
        assert sm.current_state == AnalysisState.CAUSAL

        sm.transition("causal_done")
        assert sm.current_state == AnalysisState.PREDICTIVE

        sm.transition("predictive_done")
        assert sm.current_state == AnalysisState.REPORTING

        sm.transition("report_done")
        assert sm.current_state == AnalysisState.DONE

    def test_skip_causal(self):
        """测试跳过因果分析: DESCRIPTIVE -> (skip_causal) -> PREDICTIVE"""
        sm = StateMachine()

        # 先推进到 DESCRIPTIVE 状态
        sm.transition("start")
        sm.transition("parse_done")
        assert sm.current_state == AnalysisState.DESCRIPTIVE

        # 跳过因果分析
        sm.transition("skip_causal")
        assert sm.current_state == AnalysisState.PREDICTIVE

    def test_skip_predictive(self):
        """测试跳过预测模拟: CAUSAL -> (skip_predictive) -> REPORTING"""
        sm = StateMachine()

        # 先推进到 CAUSAL 状态
        sm.transition("start")
        sm.transition("parse_done")
        sm.transition("descriptive_done")
        assert sm.current_state == AnalysisState.CAUSAL

        # 跳过预测模拟
        sm.transition("skip_predictive")
        assert sm.current_state == AnalysisState.REPORTING

    def test_error_handling(self):
        """测试错误处理: 任何非终态都可以进入 ERROR"""
        # 从 PARSING 状态进入 ERROR
        sm = StateMachine()
        sm.transition("start")
        sm.transition("error", error="解析失败")
        assert sm.current_state == AnalysisState.ERROR

        # 从 DESCRIPTIVE 状态进入 ERROR
        sm2 = StateMachine()
        sm2.transition("start")
        sm2.transition("parse_done")
        sm2.transition("error", error="描述性分析失败")
        assert sm2.current_state == AnalysisState.ERROR

        # 从 CAUSAL 状态进入 ERROR
        sm3 = StateMachine()
        sm3.transition("start")
        sm3.transition("parse_done")
        sm3.transition("descriptive_done")
        sm3.transition("error", error="因果分析失败")
        assert sm3.current_state == AnalysisState.ERROR

        # 从 PREDICTIVE 状态进入 ERROR
        sm4 = StateMachine()
        sm4.transition("start")
        sm4.transition("parse_done")
        sm4.transition("descriptive_done")
        sm4.transition("causal_done")
        sm4.transition("error", error="预测模拟失败")
        assert sm4.current_state == AnalysisState.ERROR

        # 从 REPORTING 状态进入 ERROR
        sm5 = StateMachine()
        sm5.transition("start")
        sm5.transition("parse_done")
        sm5.transition("descriptive_done")
        sm5.transition("causal_done")
        sm5.transition("predictive_done")
        sm5.transition("error", error="报告生成失败")
        assert sm5.current_state == AnalysisState.ERROR

    def test_history_logging(self):
        """测试状态转换历史记录 (StateLog)"""
        sm = StateMachine()

        # 执行几次转换
        sm.transition("start", input_summary="文件: test.csv")
        sm.transition("parse_done", output_summary="解析完成: 100 行", duration_ms=150.0)

        history = sm.get_history()
        assert len(history) == 2

        # 验证第一条日志
        log1 = history[0]
        assert log1.state == "PARSING"
        assert log1.input_summary == "文件: test.csv"
        assert log1.timestamp != ""

        # 验证第二条日志
        log2 = history[1]
        assert log2.state == "DESCRIPTIVE"
        assert log2.output_summary == "解析完成: 100 行"
        assert log2.duration_ms == 150.0

    def test_invalid_transition(self):
        """测试非法状态转换应抛出 ValueError"""
        sm = StateMachine()
        with pytest.raises(ValueError, match="非法状态转换"):
            sm.transition("parse_done")  # IDLE 状态不能直接 parse_done

    def test_reset(self):
        """测试状态机重置"""
        sm = StateMachine()
        sm.transition("start")
        sm.transition("parse_done")
        assert sm.current_state == AnalysisState.DESCRIPTIVE
        assert len(sm.get_history()) == 2

        sm.reset()
        assert sm.current_state == AnalysisState.IDLE
        assert len(sm.get_history()) == 0
