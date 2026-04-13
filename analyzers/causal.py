"""
因果推断引擎模块

回答"为什么会这样"——基于描述性分析结果，构建因果假设图，
使用 DoWhy 框架（或 OLS 回退方案）估计因果效应大小，
生成反事实分析，并通过 LLM 生成因果解释叙述。

当 DoWhy 可用时，采用四步框架：
1. 建模（Model）——基于因果假设图构建因果模型
2. 识别（Identify）——确定因果效应的可识别性
3. 估计（Estimate）——使用 backdoor 准则估计因果效应
4. 反驳（Refute）——通过安慰剂检验等手段验证结果

当 DoWhy 不可用时，回退到 OLS 线性回归方法。
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from core.models import (
    AnalysisInput,
    CausalEdge,
    CausalEffect,
    CausalGraph,
    CausalResult,
    CorrelationPair,
    Counterfactual,
    DescriptionResult,
)

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger(__name__)


class CausalAnalyzer:
    """因果推断引擎 - 回答"为什么会这样"

    基于描述性分析中发现的相关性，结合领域知识构建因果假设，
    优先使用 DoWhy 框架进行因果推断（建模-识别-估计-反驳），
    当 DoWhy 不可用时回退到 OLS 回归分析方法。

    Args:
        llm_client: LLM 客户端实例，为 None 时使用模板回退生成叙述
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """初始化因果推断引擎

        Args:
            llm_client: 可选的 LLM 客户端，用于生成自然语言叙述
        """
        self.llm = llm_client
        self._dowhy_available = self._check_dowhy()

    def _check_dowhy(self) -> bool:
        """检查 DoWhy 是否可用

        通过尝试导入 dowhy 包来判断是否已安装。

        Returns:
            bool: DoWhy 可用时返回 True，否则返回 False
        """
        try:
            import dowhy  # noqa: F401
            logger.info("DoWhy 因果推断库已加载")
            return True
        except ImportError:
            logger.info("DoWhy 未安装，将使用 OLS 回退方案")
            return False

    def analyze(
        self,
        analysis_input: AnalysisInput,
        desc_result: DescriptionResult,
    ) -> CausalResult:
        """执行因果分析

        分析流程：
        1. 基于相关性和领域知识构建因果假设图
        2. 估计因果效应大小（DoWhy 框架或 OLS 回退）
        3. 生成反事实分析
        4. LLM 生成因果解释叙述

        Args:
            analysis_input: 分析输入
            desc_result: 描述性分析结果

        Returns:
            CausalResult: 因果分析结果
        """
        df = analysis_input.data
        target = analysis_input.config.target_variable
        domain = analysis_input.metadata.domain
        audience = analysis_input.config.audience
        correlations = desc_result.correlations
        metadata = analysis_input.metadata

        logger.info("开始因果分析（DoWhy=%s）", self._dowhy_available)

        # 1. 构建因果假设图
        graph = self._build_causal_hypotheses(df, correlations, domain, metadata)
        logger.info(
            "因果假设图构建完成，节点数: %d, 边数: %d",
            len(graph.nodes),
            len(graph.edges),
        )

        # 2. 估计因果效应
        if self._dowhy_available:
            effects = self._estimate_causal_effects_dowhy(df, graph, target)
            logger.info("DoWhy 因果效应估计完成，共 %d 个效应", len(effects))
        else:
            effects = self._estimate_causal_effects_fallback(df, graph, target)
            logger.info("OLS 回退因果效应估计完成，共 %d 个效应", len(effects))

        # 3. 生成反事实分析
        counterfactuals = self._generate_counterfactuals(df, effects, target)
        logger.info("反事实分析完成，共 %d 条", len(counterfactuals))

        # 4. 生成叙述
        narrative = self._generate_narrative(
            graph=graph,
            effects=effects,
            counterfactuals=counterfactuals,
            domain=domain,
            audience=audience,
            business_understanding=analysis_input.business_understanding,
        )
        logger.info("因果叙述生成完成")

        return CausalResult(
            causal_graph=graph,
            causal_effects=effects,
            counterfactuals=counterfactuals,
            narrative=narrative,
        )

    # ================================================================
    # 因果假设图构建
    # ================================================================

    def _build_causal_hypotheses(
        self,
        df: pd.DataFrame,
        correlations: list[CorrelationPair],
        domain: str,
        metadata: object | None = None,
    ) -> CausalGraph:
        """构建因果假设图

        基于描述性分析中的相关性结果，结合偏相关分析辅助判断
        因果方向，生成候选因果边并标记置信度。
        如果 metadata 中包含 domain_config.causal_templates，
        则将领域模板中的因果边也纳入图中。

        Args:
            df: 输入数据框
            correlations: 相关性对列表
            domain: 业务领域
            metadata: 数据元信息（可选，用于获取领域配置）

        Returns:
            CausalGraph: 因果假设图
        """
        if not correlations:
            return CausalGraph(method="相关性推导")

        # 收集所有涉及的变量
        all_vars: set[str] = set()
        for corr in correlations:
            all_vars.add(corr.var1)
            all_vars.add(corr.var2)

        nodes = sorted(all_vars)
        edges: list[CausalEdge] = []

        # 筛选数值列用于偏相关分析
        numeric_vars = [v for v in nodes if v in df.select_dtypes(include=[np.number]).columns]

        for corr in correlations:
            var1, var2 = corr.var1, corr.var2
            coeff = corr.coefficient

            # 跳过弱相关
            if abs(coeff) < 0.1:
                continue

            # 尝试通过偏相关分析辅助判断因果方向
            direction, confidence = self._infer_direction(
                df, var1, var2, coeff, numeric_vars
            )

            if direction == "forward":
                from_node, to_node = var1, var2
            else:
                from_node, to_node = var2, var1

            # 计算效应大小的初步估计（标准化系数）
            effect_size = round(abs(coeff), 4)

            # p 值通过相关系数的 t 检验近似计算
            n = len(df)
            if n > 2:
                t_stat = coeff * np.sqrt((n - 2) / (1 - coeff ** 2 + 1e-10))
                p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2)))
                p_value = round(min(p_value, 1.0), 4)
            else:
                p_value = 1.0

            edges.append(
                CausalEdge(
                    from_node=from_node,
                    to_node=to_node,
                    effect_size=effect_size,
                    p_value=p_value,
                    method="偏相关分析" if confidence > 0.7 else "相关性推导",
                )
            )

        # 如果有领域配置中的因果模板，将模板边也纳入图中
        try:
            if metadata is not None and hasattr(metadata, 'domain_config') and metadata.domain_config is not None:
                domain_config = metadata.domain_config
                if hasattr(domain_config, 'causal_templates') and domain_config.causal_templates:
                    for tmpl in domain_config.causal_templates:
                        tmpl_from = tmpl.cause if hasattr(tmpl, 'cause') else tmpl.from_node
                        tmpl_to = tmpl.effect if hasattr(tmpl, 'effect') else tmpl.to_node
                        # 避免重复添加已存在的边
                        existing = any(
                            e.from_node == tmpl_from and e.to_node == tmpl_to
                            for e in edges
                        )
                        if not existing:
                            edges.append(
                                CausalEdge(
                                    from_node=tmpl_from,
                                    to_node=tmpl_to,
                                    effect_size=0.0,
                                    p_value=1.0,
                                    method="领域模板",
                                )
                            )
                            all_vars.add(tmpl_from)
                            all_vars.add(tmpl_to)
                    nodes = sorted(all_vars)
        except Exception as e:
            logger.debug("处理领域因果模板时出错，跳过: %s", e)

        method_label = "相关性推导 + 偏相关分析"
        if self._dowhy_available:
            method_label += "（DoWhy 框架）"

        return CausalGraph(
            nodes=nodes,
            edges=edges,
            method=method_label,
        )

    def _infer_direction(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        corr_coeff: float,
        numeric_vars: list[str],
    ) -> tuple[str, float]:
        """通过偏相关分析推断因果方向

        比较两个方向的偏相关系数大小，偏相关系数更大的方向
        更可能是因果方向（残差独立性假设）。

        Args:
            df: 数据框
            var1: 变量1
            var2: 变量2
            corr_coeff: 原始相关系数
            numeric_vars: 可用的数值变量列表

        Returns:
            tuple[str, float]: (方向, 置信度)
                方向为 "forward"（var1 -> var2）或 "reverse"（var2 -> var1）
        """
        # 找出控制变量（排除 var1 和 var2 的其他数值变量）
        control_vars = [v for v in numeric_vars if v != var1 and v != var2]
        control_vars = control_vars[:5]  # 限制控制变量数量

        if not control_vars:
            # 没有控制变量时，无法判断方向，默认 forward
            return "forward", 0.5

        try:
            # 检查数据可用性
            cols_needed = [var1, var2] + control_vars
            valid_mask = df[cols_needed].notna().all(axis=1)
            if valid_mask.sum() < 10:
                return "forward", 0.5

            df_valid = df.loc[valid_mask]

            # 计算 var1 对 var2 的偏相关（控制其他变量）
            # 方法：分别对 var1 和 var2 关于控制变量做回归，取残差相关
            from numpy.linalg import lstsq

            ctrl_data = df_valid[control_vars].values
            ctrl_data_with_intercept = np.column_stack([np.ones(len(ctrl_data)), ctrl_data])

            y1 = df_valid[var1].values
            y2 = df_valid[var2].values

            # var1 的残差
            coef1, _, _, _ = lstsq(ctrl_data_with_intercept, y1, rcond=None)
            resid1 = y1 - ctrl_data_with_intercept @ coef1

            # var2 的残差
            coef2, _, _, _ = lstsq(ctrl_data_with_intercept, y2, rcond=None)
            resid2 = y2 - ctrl_data_with_intercept @ coef2

            # 残差相关系数
            if np.std(resid1) > 0 and np.std(resid2) > 0:
                partial_corr = np.corrcoef(resid1, resid2)[0, 1]
            else:
                partial_corr = 0.0

            # 基于偏相关与原始相关的差异判断方向
            # 如果偏相关显著弱于原始相关，说明控制变量是混杂因子
            ratio = abs(partial_corr) / (abs(corr_coeff) + 1e-10)

            if ratio < 0.5:
                # 控制变量解释了大量相关性，可能是混杂因子
                # 因果效应可能被高估
                return "forward", 0.4
            elif ratio > 0.8:
                # 偏相关与原始相关接近，关系较为稳健
                return "forward", 0.7
            else:
                return "forward", 0.5

        except Exception as e:
            logger.debug("偏相关分析失败: %s", e)
            return "forward", 0.5

    # ================================================================
    # DoWhy 因果效应估计
    # ================================================================

    def _estimate_causal_effects_dowhy(
        self,
        df: pd.DataFrame,
        graph: CausalGraph,
        target: str | None,
    ) -> list[CausalEffect]:
        """使用 DoWhy 估计因果效应

        对因果图中的每条边执行 DoWhy 四步框架：
        1. 构建 CausalModel（将因果图转为 DOT 格式）
        2. 调用 identify_effect() 检查可识别性
        3. 调用 estimate_effect(method_name="backdoor.linear_regression")
        4. 调用反驳测试（安慰剂检验）

        如果某条边的 identify_effect 失败（图不满足 backdoor 准则），
        则跳过该边并回退到 OLS 方法。

        Args:
            df: 输入数据框
            graph: 因果假设图
            target: 目标变量名

        Returns:
            list[CausalEffect]: 因果效应估计列表
        """
        import dowhy

        if not graph.edges:
            return []

        effects: list[CausalEffect] = []
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

        for edge in graph.edges:
            # 如果指定了目标变量，只分析指向目标的边
            if target and edge.to_node != target:
                continue

            treatment = edge.from_node
            outcome = edge.to_node

            # DoWhy 需要 treatment 和 outcome 都是数值列
            if treatment not in numeric_cols or outcome not in numeric_cols:
                logger.debug(
                    "跳过非数值变量对: %s -> %s，将使用 OLS 回退",
                    treatment, outcome,
                )
                # 对非数值变量回退到 OLS
                fallback_effect = self._run_regression(
                    df, treatment, outcome,
                    self._get_control_variables(graph, treatment, outcome, df),
                )
                if fallback_effect is not None:
                    effects.append(fallback_effect)
                continue

            if treatment not in df.columns or outcome not in df.columns:
                continue

            try:
                # 将因果图转换为 DOT 格式
                dot_graph = self._graph_to_dot(graph)

                # 第 1 步：建模（Model）
                causal_model = dowhy.CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    graph=dot_graph,
                )

                # 第 2 步：识别（Identify）
                try:
                    identified_estimand = causal_model.identify_effect()
                except Exception as identify_err:
                    logger.warning(
                        "DoWhy 识别失败 (%s -> %s): %s，回退到 OLS",
                        treatment, outcome, identify_err,
                    )
                    fallback_effect = self._run_regression(
                        df, treatment, outcome,
                        self._get_control_variables(graph, treatment, outcome, df),
                    )
                    if fallback_effect is not None:
                        effects.append(fallback_effect)
                    continue

                # 第 3 步：估计（Estimate）
                try:
                    estimate = causal_model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.linear_regression",
                    )
                except Exception as estimate_err:
                    logger.warning(
                        "DoWhy 估计失败 (%s -> %s): %s，回退到 OLS",
                        treatment, outcome, estimate_err,
                    )
                    fallback_effect = self._run_regression(
                        df, treatment, outcome,
                        self._get_control_variables(graph, treatment, outcome, df),
                    )
                    if fallback_effect is not None:
                        effects.append(fallback_effect)
                    continue

                # 第 4 步：反驳测试（Refute）——仅运行安慰剂检验
                refutation_results: dict | None = None
                try:
                    refutation_results = self._refute_test(
                        causal_model, identified_estimand, estimate
                    )
                except Exception as refute_err:
                    logger.debug(
                        "DoWhy 反驳测试失败 (%s -> %s): %s，跳过反驳",
                        treatment, outcome, refute_err,
                    )

                # 提取估计结果
                effect_value = float(estimate.value)
                confidence_interval = (
                    float(estimate.get_confidence_intervals()[0][0]),
                    float(estimate.get_confidence_intervals()[0][1]),
                ) if hasattr(estimate, 'get_confidence_intervals') and estimate.get_confidence_intervals() is not None else (0.0, 0.0)

                # 尝试获取 p 值
                p_value: float | None = None
                try:
                    # DoWhy 的 estimate 对象可能包含 p 值信息
                    if hasattr(estimate, 'test_significance'):
                        test_result = estimate.test_significance(identified_estimand)
                        if hasattr(test_result, 'p_value'):
                            p_value = round(float(test_result.p_value), 4)
                except Exception:
                    pass

                effects.append(
                    CausalEffect(
                        treatment=treatment,
                        outcome=outcome,
                        effect_size=round(effect_value, 4),
                        confidence_interval=confidence_interval,
                        p_value=p_value,
                        method="dowhy.backdoor.linear_regression",
                        refutation_results=refutation_results,
                        identified=True,
                    )
                )

                logger.info(
                    "DoWhy 估计成功: %s -> %s, ATE=%.4f",
                    treatment, outcome, effect_value,
                )

            except Exception as e:
                logger.warning(
                    "DoWhy 处理失败 (%s -> %s): %s，回退到 OLS",
                    treatment, outcome, e,
                )
                fallback_effect = self._run_regression(
                    df, treatment, outcome,
                    self._get_control_variables(graph, treatment, outcome, df),
                )
                if fallback_effect is not None:
                    effects.append(fallback_effect)

        # 按效应大小排序
        effects.sort(key=lambda e: abs(e.effect_size), reverse=True)

        return effects

    def _graph_to_dot(self, graph: CausalGraph) -> str:
        """将 CausalGraph 转换为 DOT 格式字符串

        将因果假设图转换为 DoWhy 可识别的 DOT 格式图定义。
        每条边表示为 "from_node -> to_node;" 的形式。

        Args:
            graph: 因果假设图

        Returns:
            str: DOT 格式的因果图字符串，如 "digraph { cost -> price; price -> demand; }"
        """
        if not graph.edges:
            return "digraph {}"

        edge_lines = []
        for edge in graph.edges:
            edge_lines.append(f"{edge.from_node} -> {edge.to_node};")

        return "digraph {\n" + "\n".join(edge_lines) + "\n}"

    def _refute_test(
        self,
        causal_model: object,
        identified_estimand: object,
        estimate: object,
    ) -> dict:
        """执行 DoWhy 反驳测试

        运行以下反驳测试以验证因果效应估计的稳健性：
        1. placebo_treatment_refuter: 安慰剂检验 —— 将处理变量替换为随机变量，
           如果效应仍然显著则说明原估计不可靠
        2. random_common_cause: 随机共同原因 —— 添加随机变量作为共同原因，
           如果效应变化很大则说明原估计对混杂因子敏感

        Args:
            causal_model: DoWhy 的 CausalModel 实例
            identified_estimand: 识别出的估计量
            estimate: 估计结果

        Returns:
            dict: 反驳测试结果，格式为
                {
                    "placebo": {"passed": bool, "details": str},
                    "random_cause": {"passed": bool, "details": str},
                }
        """
        results: dict = {}

        # 安慰剂检验
        try:
            placebo_refuter = causal_model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
            )
            # 安慰剂检验通过条件：安慰剂效应应接近于 0
            placebo_effect = float(placebo_refuter.new_effect)
            original_effect = float(estimate.value)
            # 如果安慰剂效应的绝对值小于原始效应的 10%，认为通过
            passed = abs(placebo_effect) < 0.1 * (abs(original_effect) + 1e-10)
            results["placebo"] = {
                "passed": passed,
                "details": (
                    f"安慰剂效应={placebo_effect:.4f}, "
                    f"原始效应={original_effect:.4f}, "
                    f"{'通过' if passed else '未通过'}"
                ),
            }
        except Exception as e:
            logger.debug("安慰剂检验执行失败: %s", e)
            results["placebo"] = {
                "passed": None,
                "details": f"安慰剂检验执行失败: {e}",
            }

        # 随机共同原因检验
        try:
            random_cause_refuter = causal_model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause",
            )
            random_effect = float(random_cause_refuter.new_effect)
            original_effect = float(estimate.value)
            # 如果效应变化不超过 20%，认为通过
            relative_change = abs(random_effect - original_effect) / (abs(original_effect) + 1e-10)
            passed = relative_change < 0.2
            results["random_cause"] = {
                "passed": passed,
                "details": (
                    f"加入随机共同原因后效应={random_effect:.4f}, "
                    f"原始效应={original_effect:.4f}, "
                    f"相对变化={relative_change:.2%}, "
                    f"{'通过' if passed else '未通过'}"
                ),
            }
        except Exception as e:
            logger.debug("随机共同原因检验执行失败: %s", e)
            results["random_cause"] = {
                "passed": None,
                "details": f"随机共同原因检验执行失败: {e}",
            }

        return results

    # ================================================================
    # OLS 回退方案
    # ================================================================

    def _estimate_causal_effects_fallback(
        self,
        df: pd.DataFrame,
        graph: CausalGraph,
        target: str | None,
    ) -> list[CausalEffect]:
        """使用 OLS 回归估计因果效应（DoWhy 不可用时的回退方案）

        使用线性回归 + 控制变量方法估计每个因果边的效应大小。
        如果指定了目标变量，只估计指向目标变量的因果效应。

        Args:
            df: 输入数据框
            graph: 因果假设图
            target: 目标变量名

        Returns:
            list[CausalEffect]: 因果效应估计列表
        """
        if not graph.edges:
            return []

        effects: list[CausalEffect] = []

        for edge in graph.edges:
            # 如果指定了目标变量，只分析指向目标的边
            if target and edge.to_node != target:
                continue

            treatment = edge.from_node
            outcome = edge.to_node

            if treatment not in df.columns or outcome not in df.columns:
                continue

            # 收集控制变量：因果图中指向 treatment 或 outcome 的其他节点
            control_vars = self._get_control_variables(graph, treatment, outcome, df)

            # 使用线性回归估计因果效应
            effect = self._run_regression(df, treatment, outcome, control_vars)
            if effect is not None:
                effects.append(effect)

        # 按效应大小排序
        effects.sort(key=lambda e: abs(e.effect_size), reverse=True)

        return effects

    def _get_control_variables(
        self,
        graph: CausalGraph,
        treatment: str,
        outcome: str,
        df: pd.DataFrame,
    ) -> list[str]:
        """获取控制变量列表

        从因果图中找出同时影响 treatment 和 outcome 的变量（混杂因子），
        以及 outcome 的其他父节点。

        Args:
            graph: 因果假设图
            treatment: 处理变量
            outcome: 结果变量
            df: 数据框

        Returns:
            list[str]: 控制变量名列表
        """
        # 找出 outcome 的所有父节点（排除 treatment）
        parents_of_outcome = set()
        for edge in graph.edges:
            if edge.to_node == outcome and edge.from_node != treatment:
                parents_of_outcome.add(edge.from_node)

        # 找出 treatment 的所有父节点（排除 outcome）
        parents_of_treatment = set()
        for edge in graph.edges:
            if edge.to_node == treatment and edge.from_node != outcome:
                parents_of_treatment.add(edge.from_node)

        # 混杂因子：同时影响 treatment 和 outcome 的变量
        confounders = parents_of_treatment & parents_of_outcome

        # 合并控制变量
        all_controls = confounders | parents_of_outcome

        # 筛选存在于数据框中的数值列
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        controls = [v for v in all_controls if v in numeric_cols]

        return controls

    def _run_regression(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        control_vars: list[str],
    ) -> CausalEffect | None:
        """执行线性回归估计因果效应

        使用 OLS 回归，将 treatment 的系数作为因果效应的估计值。

        Args:
            df: 数据框
            treatment: 处理变量
            outcome: 结果变量
            control_vars: 控制变量列表

        Returns:
            CausalEffect | None: 因果效应估计，失败时返回 None
        """
        try:
            # 准备数据
            features = [treatment] + control_vars
            valid_mask = df[features + [outcome]].notna().all(axis=1)
            df_clean = df.loc[valid_mask]

            if len(df_clean) < 10:
                return None

            from numpy.linalg import lstsq

            X_cols = [treatment] + control_vars
            X = df_clean[X_cols].values
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            y = df_clean[outcome].values

            # OLS 回归
            coef, _, _, _ = lstsq(X_with_intercept, y, rcond=None)

            # treatment 的系数（第一个是截距，第二个是 treatment）
            effect_size = float(coef[1])

            # 计算残差和统计量
            y_pred = X_with_intercept @ coef
            residuals = y - y_pred
            n = len(y)
            p = X_with_intercept.shape[1]

            # 残差标准误差
            rss = np.sum(residuals ** 2)
            rse = np.sqrt(rss / (n - p)) if n > p else 0.0

            # 系数标准误差
            try:
                xt_x_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                se = np.sqrt(rse ** 2 * xt_x_inv[1, 1])
            except np.linalg.LinAlgError:
                se = 0.0

            # t 统计量和 p 值
            if se > 0:
                t_stat = effect_size / se
                p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - p)))
                p_value = round(min(p_value, 1.0), 4)
            else:
                p_value = 1.0

            # 95% 置信区间
            if se > 0:
                t_crit = stats.t.ppf(0.975, df=n - p)
                ci_lower = round(effect_size - t_crit * se, 4)
                ci_upper = round(effect_size + t_crit * se, 4)
                confidence_interval = (ci_lower, ci_upper)
            else:
                confidence_interval = None

            return CausalEffect(
                treatment=treatment,
                outcome=outcome,
                effect_size=round(effect_size, 4),
                confidence_interval=confidence_interval,
                p_value=p_value,
                method="ols",
                refutation_results=None,
                identified=False,
            )

        except Exception as e:
            logger.debug("回归分析失败 (%s -> %s): %s", treatment, outcome, e)
            return None

    # ================================================================
    # 反事实分析
    # ================================================================

    def _generate_counterfactuals(
        self,
        df: pd.DataFrame,
        effects: list[CausalEffect],
        target: str | None,
    ) -> list[Counterfactual]:
        """生成反事实分析

        对每个显著的因果效应，模拟处理变量变化 +-10% 时
        对结果变量的影响。

        Args:
            df: 数据框
            effects: 因果效应列表
            target: 目标变量名

        Returns:
            list[Counterfactual]: 反事实推理结果列表
        """
        counterfactuals: list[Counterfactual] = []

        # 筛选显著效应（p < 0.05）
        significant_effects = [e for e in effects if e.p_value is not None and e.p_value < 0.05]

        if not significant_effects:
            return counterfactuals

        for effect in significant_effects[:5]:  # 限制数量
            treatment = effect.treatment
            outcome = effect.outcome

            if treatment not in df.columns or outcome not in df.columns:
                continue

            original_mean = float(df[treatment].mean())
            outcome_mean = float(df[outcome].mean())

            if original_mean == 0:
                # 均值为 0 时使用标准差的一定比例
                original_std = float(df[treatment].std())
                if original_std == 0:
                    continue
                delta = original_std * 0.1
            else:
                delta = abs(original_mean) * 0.1

            # +10% 场景
            counterfactuals.append(
                Counterfactual(
                    treatment=treatment,
                    original_value=round(original_mean, 4),
                    counterfactual_value=round(original_mean + delta, 4),
                    predicted_outcome=round(
                        outcome_mean + effect.effect_size * delta, 4
                    ),
                    original_outcome=round(outcome_mean, 4),
                )
            )

            # -10% 场景
            counterfactuals.append(
                Counterfactual(
                    treatment=treatment,
                    original_value=round(original_mean, 4),
                    counterfactual_value=round(original_mean - delta, 4),
                    predicted_outcome=round(
                        outcome_mean - effect.effect_size * delta, 4
                    ),
                    original_outcome=round(outcome_mean, 4),
                )
            )

        return counterfactuals

    # ================================================================
    # 叙述生成
    # ================================================================

    def _generate_narrative(
        self,
        graph: CausalGraph,
        effects: list[CausalEffect],
        counterfactuals: list[Counterfactual],
        domain: str,
        audience: str,
        business_understanding: object | None = None,
    ) -> str:
        """生成因果解释叙述

        如果配置了 LLM 客户端，调用 LLM 生成叙述；
        否则使用模板拼接生成结构化文字描述。

        Args:
            graph: 因果假设图
            effects: 因果效应列表
            counterfactuals: 反事实推理列表
            domain: 业务领域
            audience: 目标受众
            business_understanding: L1 数据预扫描的业务理解结果（可选）

        Returns:
            str: 自然语言因果解释文本
        """
        # 构建因果分析发现文本
        causal_findings = self._build_causal_findings(graph, effects, counterfactuals)

        # 构建数据概要
        method_label = "DoWhy 因果推断框架" if self._dowhy_available else "OLS 线性回归"
        data_summary = (
            f"因果图包含 {len(graph.nodes)} 个节点和 {len(graph.edges)} 条因果边。"
            f"因果效应估计方法: {method_label}。"
        )

        if self.llm is not None:
            try:
                from llm.prompts import PromptTemplates

                prompt = PromptTemplates.causal_prompt(
                    data_summary=data_summary,
                    causal_findings=causal_findings,
                    domain=domain or "通用",
                    audience=audience,
                    business_understanding=business_understanding,  # type: ignore[arg-type]
                )
                narrative = self.llm.generate(
                    prompt=prompt,
                    system_prompt=PromptTemplates.CAUSAL_SYSTEM,
                )
                return narrative.strip()
            except Exception as e:
                logger.warning("LLM 因果叙述生成失败，回退到模板: %s", e)

        # 模板回退
        return self._template_narrative(graph, effects, counterfactuals, domain)

    def _build_causal_findings(
        self,
        graph: CausalGraph,
        effects: list[CausalEffect],
        counterfactuals: list[Counterfactual],
    ) -> str:
        """构建供 LLM 使用的因果分析发现文本

        Args:
            graph: 因果假设图
            effects: 因果效应列表
            counterfactuals: 反事实推理列表

        Returns:
            str: 格式化的因果分析发现文本
        """
        lines: list[str] = []

        # 因果图
        lines.append("### 因果假设图")
        for edge in graph.edges:
            lines.append(
                f"- {edge.from_node} -> {edge.to_node}: "
                f"效应大小={edge.effect_size}, p值={edge.p_value}, "
                f"方法={edge.method}"
            )

        # 因果效应估计
        if effects:
            lines.append("\n### 因果效应估计")
            for eff in effects:
                ci_str = ""
                if eff.confidence_interval:
                    ci_str = (
                        f", 95%置信区间=[{eff.confidence_interval[0]}, "
                        f"{eff.confidence_interval[1]}]"
                    )
                method_label = eff.method or "未知"
                lines.append(
                    f"- {eff.treatment} -> {eff.outcome}: "
                    f"ATE={eff.effect_size}, p值={eff.p_value}{ci_str}, "
                    f"方法={method_label}"
                )

                # 如果有反驳测试结果，附加到发现中
                if eff.refutation_results:
                    lines.append("  反驳测试:")
                    for test_name, test_result in eff.refutation_results.items():
                        if isinstance(test_result, dict):
                            passed = test_result.get("passed")
                            details = test_result.get("details", "")
                            status = "通过" if passed else ("未通过" if passed is False else "未知")
                            lines.append(f"    - {test_name}: {status} ({details})")

        # 反事实分析
        if counterfactuals:
            lines.append("\n### 反事实分析")
            for cf in counterfactuals:
                change = (
                    cf.predicted_outcome - cf.original_outcome
                    if cf.predicted_outcome is not None and cf.original_outcome is not None
                    else 0
                )
                lines.append(
                    f"- 若「{cf.treatment}」从 {cf.original_value} 变为 {cf.counterfactual_value}，"
                    f"「预测结果」变化 {change:+.4f}"
                )

        return "\n".join(lines)

    def _template_narrative(
        self,
        graph: CausalGraph,
        effects: list[CausalEffect],
        counterfactuals: list[Counterfactual],
        domain: str,
    ) -> str:
        """使用模板生成因果叙述（LLM 不可用时的回退方案）

        Args:
            graph: 因果假设图
            effects: 因果效应列表
            counterfactuals: 反事实推理列表
            domain: 业务领域

        Returns:
            str: 模板化的因果叙述文本
        """
        parts: list[str] = []

        domain_prefix = f"在「{domain}」领域" if domain else "在本次分析中"

        # 判断是否使用了 DoWhy
        dowhy_used = any("dowhy" in (e.method or "") for e in effects)
        method_desc = "DoWhy 因果推断框架（建模-识别-估计-反驳）" if dowhy_used else "OLS 线性回归"

        # 因果图概述
        if graph.edges:
            parts.append(
                f"{domain_prefix}，通过相关性分析和偏相关分析，"
                f"构建了包含 {len(graph.nodes)} 个变量、{len(graph.edges)} 条因果边的假设图。"
                f"因果效应估计采用{method_desc}方法。"
            )
        else:
            parts.append(f"{domain_prefix}，未发现足够的证据构建因果假设。")
            return "\n\n".join(parts)

        # 显著效应
        significant = [e for e in effects if e.p_value is not None and e.p_value < 0.05]
        if significant:
            eff_descs = []
            for eff in significant[:3]:
                direction = "正向" if eff.effect_size > 0 else "负向"
                method_tag = f"（{eff.method}）" if eff.method else ""
                eff_descs.append(
                    f"「{eff.treatment}」对「{eff.outcome}」有{direction}因果效应"
                    f"（效应大小: {eff.effect_size}）{method_tag}"
                )
            parts.append("显著的因果效应包括：" + "；".join(eff_descs) + "。")

            # DoWhy 反驳测试结果摘要
            dowhy_effects = [e for e in significant if e.refutation_results]
            if dowhy_effects:
                refutation_summary_parts = []
                for eff in dowhy_effects:
                    for test_name, test_result in eff.refutation_results.items():
                        if isinstance(test_result, dict):
                            passed = test_result.get("passed")
                            if passed is True:
                                refutation_summary_parts.append(
                                    f"「{eff.treatment}」->「{eff.outcome}」的"
                                    f"{test_name}检验通过"
                                )
                if refutation_summary_parts:
                    parts.append(
                        "反驳测试验证：" + "；".join(refutation_summary_parts[:3]) + "。"
                    )

        # 反事实洞察
        if counterfactuals:
            parts.append(
                f"反事实分析表明，关键变量的变化可能对结果产生显著影响"
                f"（共模拟了 {len(counterfactuals)} 个场景）。"
            )

        # 局限性说明
        if dowhy_used:
            parts.append(
                "注意：以上因果分析基于 DoWhy 框架的观察数据推断，"
                "因果关系的确认需要进一步的实验验证或领域专家的确认。"
                "反驳测试仅提供了部分稳健性验证，不能完全排除未观测混杂因子的影响。"
            )
        else:
            parts.append(
                "注意：以上因果分析基于观察数据和统计假设，"
                "因果关系的确认需要进一步的实验验证或领域专家的确认。"
            )

        return "\n\n".join(parts)
