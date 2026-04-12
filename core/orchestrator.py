"""
编排器模块

负责协调整个分析流程，按顺序调用解析、描述性分析、因果分析、
预测模拟和报告生成等阶段，并通过状态机管理生命周期。
支持断点续跑和异常处理。

在解析完成后、描述性分析之前，可选地执行数据预扫描步骤，
让 LLM 理解数据的业务语义，生成 BusinessUnderstanding。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.models import (
    AnalysisConfig,
    AnalysisInput,
    CausalResult,
    DescriptionResult,
    ExplainabilityReport,
    PredictionResult,
)
from core.state_machine import AnalysisState, StateMachine

logger = logging.getLogger(__name__)


class Orchestrator:
    """分析编排器

    协调各分析阶段的执行顺序，管理状态转换并记录日志。
    支持断点续跑：如果状态机已有中间结果，可从上次中断处继续。

    分析流程：
    1. 解析数据文件（PARSING 状态）
    2. 数据预扫描（可选，作为 PARSING 的一部分，生成 BusinessUnderstanding）
    3. 描述性分析（DESCRIPTIVE 状态）
    4. 因果分析（可选，CAUSAL 状态）
    5. 预测模拟（可选，PREDICTIVE 状态）
    6. 报告生成（REPORTING 状态）

    Attributes:
        config: 分析配置
        llm_client: 大语言模型客户端
        rules_engine: 规则引擎
        state_machine: 状态机实例
        _cached_input: 缓存的解析结果（用于断点续跑）
        _cached_desc: 缓存的描述性分析结果（用于断点续跑）
        _cached_causal: 缓存的因果分析结果（用于断点续跑）
        _cached_pred: 缓存的预测模拟结果（用于断点续跑）
    """

    def __init__(
        self,
        config: AnalysisConfig,
        llm_client: Any = None,
        rules_engine: Any = None,
    ) -> None:
        """初始化编排器

        Args:
            config: 分析配置
            llm_client: 大语言模型客户端（可选）
            rules_engine: 规则引擎实例（可选）
        """
        self.config = config
        self.llm_client = llm_client
        self.rules_engine = rules_engine
        self.state_machine = StateMachine()

        # 缓存中间结果，用于断点续跑
        self._cached_input: AnalysisInput | None = None
        self._cached_desc: DescriptionResult | None = None
        self._cached_causal: CausalResult | None = None
        self._cached_pred: PredictionResult | None = None

    def run(self, file_path: str) -> ExplainabilityReport:
        """执行完整的分析流程

        按照状态机定义的流程，依次执行解析、数据预扫描（可选）、
        描述性分析、因果分析（可选）、预测模拟（可选）和报告生成。
        支持断点续跑：如果状态机已有中间结果，可从上次中断处继续。

        Args:
            file_path: 数据文件路径

        Returns:
            完整的可解释性分析报告

        Raises:
            Exception: 分析过程中发生错误时抛出
        """
        try:
            # 检查是否需要从断点续跑
            current_state = self.state_machine.current_state

            # 阶段 1: 解析数据
            if current_state == AnalysisState.IDLE:
                analysis_input = self._parse(file_path)
                self._cached_input = analysis_input
            elif self._cached_input is not None:
                logger.info("断点续跑：跳过解析阶段，使用缓存数据")
                analysis_input = self._cached_input
            else:
                # 状态机不在 IDLE 但没有缓存，重新解析
                logger.warning("状态机不在 IDLE 但无缓存数据，重新解析")
                self.state_machine.reset()
                analysis_input = self._parse(file_path)
                self._cached_input = analysis_input

            # 阶段 1.5: 数据预扫描（解析完成后、描述性分析之前）
            # 将扫描作为 parsing 阶段的一部分，不新增状态
            if self.llm_client is not None:
                analysis_input = self._scan_business_context(analysis_input)

            # 阶段 2: 描述性分析
            current_state = self.state_machine.current_state
            if current_state == AnalysisState.PARSING:
                desc_result = self._analyze_descriptive(analysis_input)
                self._cached_desc = desc_result
            elif self._cached_desc is not None:
                logger.info("断点续跑：跳过描述性分析阶段，使用缓存结果")
                desc_result = self._cached_desc
            else:
                logger.warning("状态异常，重新执行描述性分析")
                desc_result = self._analyze_descriptive(analysis_input)
                self._cached_desc = desc_result

            # 阶段 3: 因果分析（可选）
            causal_result: CausalResult | None = None
            if self.config.causal_enabled:
                current_state = self.state_machine.current_state

                # 使用规则引擎判断是否应跳过因果分析
                should_skip = self._should_skip_causal(analysis_input)

                if should_skip:
                    logger.info("规则引擎建议跳过因果分析: %s", should_skip)
                    self.state_machine.transition(
                        "skip_causal",
                        input_summary=f"因果分析已跳过（规则引擎: {should_skip}）",
                    )
                elif current_state == AnalysisState.DESCRIPTIVE:
                    causal_result = self._analyze_causal(analysis_input, desc_result)
                    self._cached_causal = causal_result
                elif self._cached_causal is not None:
                    logger.info("断点续跑：跳过因果分析阶段，使用缓存结果")
                    causal_result = self._cached_causal
                else:
                    causal_result = self._analyze_causal(analysis_input, desc_result)
                    self._cached_causal = causal_result
            else:
                self.state_machine.transition(
                    "skip_causal",
                    input_summary="因果分析已跳过（配置禁用）",
                )

            # 阶段 4: 预测模拟（可选）
            pred_result: PredictionResult | None = None
            if self.config.predictive_enabled:
                current_state = self.state_machine.current_state

                if current_state in (AnalysisState.CAUSAL, AnalysisState.DESCRIPTIVE):
                    pred_result = self._simulate_predictive(
                        analysis_input, desc_result, causal_result
                    )
                    self._cached_pred = pred_result
                elif self._cached_pred is not None:
                    logger.info("断点续跑：跳过预测模拟阶段，使用缓存结果")
                    pred_result = self._cached_pred
                else:
                    pred_result = self._simulate_predictive(
                        analysis_input, desc_result, causal_result
                    )
                    self._cached_pred = pred_result
            else:
                self.state_machine.transition(
                    "skip_predictive",
                    input_summary="预测模拟已跳过（配置禁用）",
                )

            # 阶段 5: 生成报告
            report = self._generate_report(
                analysis_input, desc_result, causal_result, pred_result
            )

            # 清理缓存
            self._clear_cache()

            return report

        except Exception as exc:
            logger.error("分析流程异常: %s", exc, exc_info=True)
            # 尝试将状态机转入 ERROR 状态
            try:
                self.state_machine.transition("error", error=str(exc))
            except ValueError:
                pass  # 如果当前状态无法转入 ERROR（如已在终态），忽略
            raise

    def _scan_business_context(self, analysis_input: AnalysisInput) -> AnalysisInput:
        """数据预扫描：让 LLM 理解数据业务语义

        在解析完成后、描述性分析之前执行。使用 DataScanner
        分析数据的列名、类型和统计特征，推断业务场景和关键指标，
        生成 BusinessUnderstanding 对象并附加到 analysis_input 上。

        如果预扫描失败（例如 LLM 不可用或超时），会静默跳过，
        不影响后续分析流程。

        Args:
            analysis_input: 已解析的分析输入

        Returns:
            AnalysisInput: 附带了 business_understanding 的分析输入
        """
        try:
            from llm.data_scanner import DataScanner

            scanner = DataScanner(llm_client=self.llm_client)
            understanding = scanner.scan(analysis_input)
            analysis_input.business_understanding = understanding
            logger.info("数据预扫描完成，业务场景: %s", understanding.inferred_scenario if understanding else "无")
        except ImportError:
            logger.debug("DataScanner 模块不可用，跳过数据预扫描")
        except Exception as e:
            logger.warning("数据预扫描失败，跳过: %s", e)

        return analysis_input

    def _should_skip_causal(self, analysis_input: AnalysisInput) -> str | None:
        """使用规则引擎判断是否应跳过因果分析

        评估流程控制规则，如果规则建议跳过因果分析则返回原因，
        否则返回 None。

        Args:
            analysis_input: 分析输入

        Returns:
            str | None: 跳过原因，不跳过时返回 None
        """
        if self.rules_engine is None:
            return None

        context = {
            "variable_count": len(analysis_input.metadata.columns),
            "row_count": analysis_input.metadata.row_count,
            "missing_ratio": analysis_input.metadata.missing_ratio,
        }

        try:
            triggered_actions = self.rules_engine.get_triggered_actions(context)
            for rule_name, action_result in triggered_actions:
                if isinstance(action_result, dict):
                    target = action_result.get("target", "")
                    if target == "causal_inference":
                        return action_result.get("reason", "规则引擎建议跳过")
        except Exception as e:
            logger.warning("规则引擎评估失败: %s", e)

        return None

    def _parse(self, file_path: str) -> AnalysisInput:
        """解析数据文件

        读取文件并构建 AnalysisInput 对象。
        根据文件扩展名自动选择解析器（CSV, Excel, JSON 等），
        自动推断列类型并构建 DataMetadata。

        Args:
            file_path: 数据文件路径

        Returns:
            解析后的分析输入

        Raises:
            ValueError: 文件不存在或格式不支持
        """
        self.state_machine.transition("start", input_summary=f"文件: {file_path}")
        start = time.perf_counter()

        try:
            from analyzers.parser import FileParser

            parser = FileParser()
            analysis_input = parser.parse(file_path, config=self.config)

            duration_ms = (time.perf_counter() - start) * 1000
            self.state_machine.transition(
                "parse_done",
                input_summary=f"解析完成: {analysis_input.metadata.row_count} 行, "
                              f"{len(analysis_input.metadata.columns)} 列",
                duration_ms=duration_ms,
            )

            logger.info(
                "数据解析完成: %d 行, %d 列, 缺失率: %.2f%%",
                analysis_input.metadata.row_count,
                len(analysis_input.metadata.columns),
                analysis_input.metadata.missing_ratio * 100,
            )

            return analysis_input

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("数据解析失败: %s", exc)
            self.state_machine.transition(
                "error",
                input_summary=f"解析失败: {file_path}",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

    def _analyze_descriptive(self, input_data: AnalysisInput) -> DescriptionResult:
        """执行描述性分析

        计算变量重要性、分布统计、相关性和异常检测。

        Args:
            input_data: 分析输入

        Returns:
            描述性分析结果
        """
        start = time.perf_counter()

        try:
            from analyzers.descriptive import DescriptiveAnalyzer

            analyzer = DescriptiveAnalyzer(llm_client=self.llm_client)
            result = analyzer.analyze(input_data)

            duration_ms = (time.perf_counter() - start) * 1000
            self.state_machine.transition(
                "descriptive_done",
                input_summary=f"描述性分析完成",
                output_summary=f"变量重要性: {len(result.variable_importance)} 个, "
                              f"相关性: {len(result.correlations)} 对, "
                              f"异常值: {len(result.anomalies)} 个",
                duration_ms=duration_ms,
            )

            return result

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("描述性分析失败: %s", exc)
            self.state_machine.transition(
                "error",
                input_summary="描述性分析失败",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

    def _analyze_causal(
        self,
        input_data: AnalysisInput,
        desc_result: DescriptionResult,
    ) -> CausalResult:
        """执行因果分析

        基于描述性分析结果，进行因果发现和效应估计。

        Args:
            input_data: 分析输入
            desc_result: 描述性分析结果

        Returns:
            因果分析结果
        """
        start = time.perf_counter()

        try:
            from analyzers.causal import CausalAnalyzer

            analyzer = CausalAnalyzer(llm_client=self.llm_client)
            result = analyzer.analyze(input_data, desc_result)

            duration_ms = (time.perf_counter() - start) * 1000
            edge_count = (
                len(result.causal_graph.edges) if result.causal_graph else 0
            )
            self.state_machine.transition(
                "causal_done",
                input_summary=f"因果分析完成",
                output_summary=f"因果边: {edge_count} 条, "
                              f"效应: {len(result.causal_effects)} 个, "
                              f"反事实: {len(result.counterfactuals)} 条",
                duration_ms=duration_ms,
            )

            return result

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("因果分析失败: %s", exc)
            self.state_machine.transition(
                "error",
                input_summary="因果分析失败",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

    def _simulate_predictive(
        self,
        input_data: AnalysisInput,
        desc_result: DescriptionResult,
        causal_result: CausalResult | None,
    ) -> PredictionResult:
        """执行预测模拟

        基于描述性和因果分析结果，进行场景模拟和敏感性分析。

        Args:
            input_data: 分析输入
            desc_result: 描述性分析结果
            causal_result: 因果分析结果（可选）

        Returns:
            预测模拟结果
        """
        start = time.perf_counter()

        try:
            from analyzers.predictive import PredictiveSimulator

            simulator = PredictiveSimulator(llm_client=self.llm_client)
            result = simulator.simulate(input_data, desc_result, causal_result)

            duration_ms = (time.perf_counter() - start) * 1000
            self.state_machine.transition(
                "predictive_done",
                input_summary=f"预测模拟完成",
                output_summary=f"场景: {len(result.scenarios)} 个, "
                              f"What-If: {len(result.what_ifs)} 条",
                duration_ms=duration_ms,
            )

            return result

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("预测模拟失败: %s", exc)
            self.state_machine.transition(
                "error",
                input_summary="预测模拟失败",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

    def _generate_report(
        self,
        input_data: AnalysisInput,
        desc_result: DescriptionResult,
        causal_result: CausalResult | None,
        pred_result: PredictionResult | None,
    ) -> ExplainabilityReport:
        """生成可解释性报告

        汇总所有分析结果，调用 ReportGenerator 生成自然语言叙述，
        并组装最终报告。business_understanding 会通过 input_data
        自动传递给报告生成器。

        Args:
            input_data: 分析输入（包含 business_understanding）
            desc_result: 描述性分析结果
            causal_result: 因果分析结果（可选）
            pred_result: 预测模拟结果（可选）

        Returns:
            完整的可解释性分析报告
        """
        start = time.perf_counter()

        try:
            from report.generator import ReportGenerator

            generator = ReportGenerator(llm_client=self.llm_client)
            report = generator.generate(
                analysis_input=input_data,
                desc_result=desc_result,
                causal_result=causal_result,
                pred_result=pred_result,
            )

            duration_ms = (time.perf_counter() - start) * 1000
            self.state_machine.transition(
                "report_done",
                input_summary="报告生成完成",
                output_summary=f"图表: {len(report.charts)} 个",
                duration_ms=duration_ms,
            )

            logger.info("报告生成完成，耗时 %.1f ms", duration_ms)

            return report

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("报告生成失败: %s", exc)
            self.state_machine.transition(
                "error",
                input_summary="报告生成失败",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

    def _clear_cache(self) -> None:
        """清理缓存的中间结果"""
        self._cached_input = None
        self._cached_desc = None
        self._cached_causal = None
        self._cached_pred = None

    def get_state_history(self) -> list:
        """获取状态转换历史

        Returns:
            list[StateLog]: 状态日志列表
        """
        return self.state_machine.get_history()
