"""
描述性分析器模块

回答"发生了什么"——对数据进行全面的描述性统计分析，
包括变量重要性排名、分布特征、相关性分析和异常值检测。
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
    Anomaly,
    CorrelationPair,
    DescriptionResult,
    Distribution,
    VarImportance,
)

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger(__name__)


class DescriptiveAnalyzer:
    """描述性分析器 - 回答"发生了什么"

    对输入数据进行全面的描述性统计分析，涵盖变量重要性、
    分布特征、变量间相关性以及异常值检测，并支持通过
    LLM 生成自然语言叙述。

    Args:
        llm_client: LLM 客户端实例，为 None 时使用模板回退生成叙述
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """初始化描述性分析器

        Args:
            llm_client: 可选的 LLM 客户端，用于生成自然语言叙述
        """
        self.llm = llm_client

    def analyze(self, analysis_input: AnalysisInput) -> DescriptionResult:
        """执行完整的描述性分析

        分析流程：
        1. 计算变量重要性（基于方差贡献或互信息）
        2. 计算各数值变量的分布特征
        3. 计算变量间相关性矩阵
        4. 检测异常值（IQR 方法）
        5. 生成自然语言叙述（LLM 或模板回退）

        Args:
            analysis_input: 分析输入，包含数据、元信息和配置

        Returns:
            DescriptionResult: 描述性分析结果
        """
        df = analysis_input.data
        columns = analysis_input.metadata.columns
        target = analysis_input.config.target_variable
        domain = analysis_input.metadata.domain
        audience = analysis_input.config.audience

        logger.info("开始描述性分析，数据形状: %s", df.shape)

        # 1. 计算变量重要性
        importance = self._compute_variable_importance(df, target)
        logger.info("变量重要性计算完成，共 %d 个变量", len(importance))

        # 2. 计算各变量分布特征
        distributions = self._compute_distributions(df, columns)
        logger.info("分布特征计算完成，共 %d 个数值变量", len(distributions))

        # 3. 计算相关性矩阵
        correlations = self._compute_correlations(df, columns)
        logger.info("相关性计算完成，发现 %d 对显著相关", len(correlations))

        # 4. 检测异常值
        anomalies = self._detect_anomalies(df, columns)
        logger.info("异常值检测完成，发现 %d 个异常值", len(anomalies))

        # 5. 生成自然语言叙述
        narrative = self._generate_narrative(
            importance=importance,
            distributions=distributions,
            correlations=correlations,
            anomalies=anomalies,
            domain=domain,
            audience=audience,
        )
        logger.info("叙述生成完成")

        return DescriptionResult(
            variable_importance=importance,
            distributions=list(distributions.values()),
            correlations=correlations,
            anomalies=anomalies,
            narrative=narrative,
        )

    def _compute_variable_importance(
        self, df: pd.DataFrame, target: str | None
    ) -> list[VarImportance]:
        """计算变量重要性排名

        如果指定了目标变量，使用互信息（mutual information）衡量各变量
        对目标变量的预测能力；否则使用方差贡献比例作为重要性指标。

        Args:
            df: 输入数据框
            target: 目标变量名，为 None 时使用方差贡献

        Returns:
            list[VarImportance]: 按重要性降序排列的变量重要性列表
        """
        # 筛选数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return []

        if target and target in df.columns:
            # 有目标变量：使用互信息
            return self._importance_by_mutual_info(df, numeric_cols, target)
        else:
            # 无目标变量：使用方差贡献比例
            return self._importance_by_variance(df, numeric_cols)

    def _importance_by_mutual_info(
        self, df: pd.DataFrame, numeric_cols: list[str], target: str
    ) -> list[VarImportance]:
        """基于互信息计算变量重要性

        对连续目标变量使用 mutual_info_regression，对离散目标变量
        使用 mutual_info_classif。

        Args:
            df: 输入数据框
            numeric_cols: 数值列名列表
            target: 目标变量名

        Returns:
            list[VarImportance]: 变量重要性列表
        """
        from sklearn.feature_selection import mutual_info_regression

        # 准备特征和目标
        feature_cols = [c for c in numeric_cols if c != target]
        if not feature_cols:
            return []

        # 去除缺失值
        valid_mask = df[feature_cols + [target]].notna().all(axis=1)
        df_clean = df.loc[valid_mask]

        if len(df_clean) < 10:
            logger.warning("有效样本不足 10 条，回退到方差贡献方法")
            return self._importance_by_variance(df, numeric_cols)

        X = df_clean[feature_cols].values
        y = df_clean[target].values

        # 计算互信息
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_scores = mutual_info_regression(X, y, random_state=42)

        # 归一化
        total = mi_scores.sum()
        if total > 0:
            mi_scores = mi_scores / total

        # 构建结果并排序
        results = []
        for col, score in zip(feature_cols, mi_scores):
            results.append(VarImportance(name=col, score=round(float(score), 4), rank=0))

        results.sort(key=lambda x: x.score, reverse=True)

        # 分配排名
        for i, item in enumerate(results):
            item.rank = i + 1

        return results

    def _importance_by_variance(
        self, df: pd.DataFrame, numeric_cols: list[str]
    ) -> list[VarImportance]:
        """基于方差贡献比例计算变量重要性

        使用各变量方差占总方差的比例作为重要性指标。

        Args:
            df: 输入数据框
            numeric_cols: 数值列名列表

        Returns:
            list[VarImportance]: 变量重要性列表
        """
        variances = df[numeric_cols].var().fillna(0)
        total_var = variances.sum()

        if total_var == 0:
            return [
                VarImportance(name=col, score=0.0, rank=i + 1)
                for i, col in enumerate(numeric_cols)
            ]

        proportions = variances / total_var

        results = []
        for col in numeric_cols:
            score = round(float(proportions[col]), 4)
            results.append(VarImportance(name=col, score=score, rank=0))

        results.sort(key=lambda x: x.score, reverse=True)

        for i, item in enumerate(results):
            item.rank = i + 1

        return results

    def _compute_distributions(
        self, df: pd.DataFrame, columns: list
    ) -> dict[str, Distribution]:
        """计算各数值变量的分布特征

        对每个数值列计算均值、中位数、标准差、偏度、峰度、
        最小值、最大值，并用 IQR 方法统计异常值数量。

        Args:
            df: 输入数据框
            columns: 列元信息列表

        Returns:
            dict[str, Distribution]: 列名到分布统计的映射
        """
        distributions: dict[str, Distribution] = {}

        for col_meta in columns:
            if col_meta.type != "numeric":
                continue

            col_name = col_meta.name
            if col_name not in df.columns:
                continue

            series = df[col_name].dropna()
            if len(series) == 0:
                distributions[col_name] = Distribution(column=col_name)
                continue

            # 计算基本统计量
            dist = Distribution(
                column=col_name,
                mean=round(float(series.mean()), 4),
                median=round(float(series.median()), 4),
                std=round(float(series.std()), 4),
                skewness=round(float(series.skew()), 4),
                kurtosis=round(float(series.kurtosis()), 4),
                min=round(float(series.min()), 4),
                max=round(float(series.max()), 4),
            )

            # IQR 异常值统计
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outlier_ratio = round(outlier_count / len(series), 4)

            dist.outlier_count = outlier_count
            dist.outlier_ratio = outlier_ratio

            distributions[col_name] = dist

        return distributions

    def _compute_correlations(
        self, df: pd.DataFrame, columns: list
    ) -> list[CorrelationPair]:
        """计算变量间相关性

        两个数值列之间使用 Pearson 相关系数；涉及分类列时
        使用 Spearman 秩相关系数。仅返回绝对值大于 0.1 的结果。

        Args:
            df: 输入数据框
            columns: 列元信息列表

        Returns:
            list[CorrelationPair]: 显著相关性对列表
        """
        results: list[CorrelationPair] = []

        # 区分数值列和分类列
        numeric_names = [c.name for c in columns if c.type == "numeric" and c.name in df.columns]
        categorical_names = [c.name for c in columns if c.type == "categorical" and c.name in df.columns]

        # 数值列之间的 Pearson 相关
        if len(numeric_names) >= 2:
            corr_matrix = df[numeric_names].corr(method="pearson")
            for i, col1 in enumerate(numeric_names):
                for col2 in numeric_names[i + 1:]:
                    coeff = corr_matrix.loc[col1, col2]
                    if not np.isnan(coeff) and abs(coeff) > 0.1:
                        results.append(
                            CorrelationPair(
                                var1=col1,
                                var2=col2,
                                coefficient=round(float(coeff), 4),
                                method="pearson",
                            )
                        )

        # 涉及分类列的 Spearman 相关
        all_cols_for_spearman = numeric_names + categorical_names
        if len(all_cols_for_spearman) >= 2:
            # 对包含分类列的列对使用 Spearman
            for i, col1 in enumerate(all_cols_for_spearman):
                for col2 in all_cols_for_spearman[i + 1:]:
                    # 如果两个都是数值列，已经用 Pearson 算过了，跳过
                    if col1 in numeric_names and col2 in numeric_names:
                        continue

                    try:
                        # Spearman 需要非空且可排序的值
                        valid_mask = df[[col1, col2]].notna().all(axis=1)
                        if valid_mask.sum() < 5:
                            continue

                        coeff, _ = stats.spearmanr(
                            df.loc[valid_mask, col1],
                            df.loc[valid_mask, col2],
                        )
                        if not np.isnan(coeff) and abs(coeff) > 0.1:
                            results.append(
                                CorrelationPair(
                                    var1=col1,
                                    var2=col2,
                                    coefficient=round(float(coeff), 4),
                                    method="spearman",
                                )
                            )
                    except (ValueError, TypeError):
                        continue

        # 按相关系数绝对值降序排列
        results.sort(key=lambda x: abs(x.coefficient), reverse=True)

        return results

    def _detect_anomalies(
        self, df: pd.DataFrame, columns: list
    ) -> list[Anomaly]:
        """使用 IQR 方法检测异常值

        对每个数值列，将超出 [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 范围的
        值标记为异常值。

        Args:
            df: 输入数据框
            columns: 列元信息列表

        Returns:
            list[Anomaly]: 异常记录列表
        """
        anomalies: list[Anomaly] = []

        for col_meta in columns:
            if col_meta.type != "numeric":
                continue

            col_name = col_meta.name
            if col_name not in df.columns:
                continue

            series = df[col_name]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # 找出异常值
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outlier_mask].index

            for idx in outlier_indices:
                anomalies.append(
                    Anomaly(
                        column=col_name,
                        value=series.loc[idx],
                        index=int(idx),
                        type="outlier",
                    )
                )

        return anomalies

    def _generate_narrative(
        self,
        importance: list[VarImportance],
        distributions: dict[str, Distribution],
        correlations: list[CorrelationPair],
        anomalies: list[Anomaly],
        domain: str,
        audience: str,
    ) -> str:
        """生成自然语言叙述

        如果配置了 LLM 客户端，调用 LLM 生成叙述；
        否则使用模板拼接生成结构化文字描述。

        Args:
            importance: 变量重要性列表
            distributions: 分布统计字典
            correlations: 相关性对列表
            anomalies: 异常值列表
            domain: 业务领域
            audience: 目标受众

        Returns:
            str: 自然语言叙述文本
        """
        # 构建数据概要
        data_summary = self._build_data_summary(
            importance, distributions, correlations, anomalies
        )

        if self.llm is not None:
            try:
                from llm.prompts import PromptTemplates

                prompt = PromptTemplates.descriptive_prompt(
                    data_summary=data_summary,
                    domain=domain or "通用",
                    audience=audience,
                )
                narrative = self.llm.generate(
                    prompt=prompt,
                    system_prompt=PromptTemplates.DESCRIPTIVE_SYSTEM,
                )
                return narrative.strip()
            except Exception as e:
                logger.warning("LLM 叙述生成失败，回退到模板: %s", e)

        # 模板回退
        return self._template_narrative(
            importance, distributions, correlations, anomalies, domain
        )

    def _build_data_summary(
        self,
        importance: list[VarImportance],
        distributions: dict[str, Distribution],
        correlations: list[CorrelationPair],
        anomalies: list[Anomaly],
    ) -> str:
        """构建供 LLM 使用的数据概要文本

        Args:
            importance: 变量重要性列表
            distributions: 分布统计字典
            correlations: 相关性对列表
            anomalies: 异常值列表

        Returns:
            str: 格式化的数据概要文本
        """
        lines: list[str] = []

        # 变量重要性
        if importance:
            lines.append("### 变量重要性排名")
            for item in importance[:10]:
                lines.append(f"- {item.name}: 重要性得分 {item.score}（排名 #{item.rank}）")

        # 分布特征
        if distributions:
            lines.append("\n### 变量分布特征")
            for col_name, dist in distributions.items():
                lines.append(
                    f"- {col_name}: 均值={dist.mean}, 中位数={dist.median}, "
                    f"标准差={dist.std}, 偏度={dist.skewness}, 峰度={dist.kurtosis}, "
                    f"范围=[{dist.min}, {dist.max}], 异常值数量={dist.outlier_count}"
                )

        # 相关性
        if correlations:
            lines.append("\n### 重要相关性")
            for corr in correlations[:10]:
                direction = "正相关" if corr.coefficient > 0 else "负相关"
                lines.append(
                    f"- {corr.var1} 与 {corr.var2}: "
                    f"{direction}（{corr.method}, r={corr.coefficient}）"
                )

        # 异常值
        if anomalies:
            lines.append(f"\n### 异常值（共 {len(anomalies)} 个）")
            # 按列汇总
            from collections import Counter

            col_counts = Counter(a.column for a in anomalies)
            for col_name, count in col_counts.most_common(10):
                lines.append(f"- {col_name}: {count} 个异常值")

        return "\n".join(lines)

    def _template_narrative(
        self,
        importance: list[VarImportance],
        distributions: dict[str, Distribution],
        correlations: list[CorrelationPair],
        anomalies: list[Anomaly],
        domain: str,
    ) -> str:
        """使用模板生成叙述（LLM 不可用时的回退方案）

        Args:
            importance: 变量重要性列表
            distributions: 分布统计字典
            correlations: 相关性对列表
            anomalies: 异常值列表
            domain: 业务领域

        Returns:
            str: 模板化的叙述文本
        """
        parts: list[str] = []

        domain_prefix = f"在「{domain}」领域的数据分析中" if domain else "在本次数据分析中"

        # 概况
        parts.append(f"{domain_prefix}，共分析了 {len(distributions)} 个数值变量。")

        # 重要性
        if importance:
            top3 = importance[:3]
            top_names = "、".join(f"「{item.name}」（得分 {item.score}）" for item in top3)
            parts.append(f"最重要的变量为 {top_names}。")

        # 分布
        if distributions:
            high_skew = [
                (name, dist) for name, dist in distributions.items()
                if dist.skewness is not None and abs(dist.skewness) > 1
            ]
            if high_skew:
                skew_names = "、".join(f"「{name}」（偏度 {dist.skewness}）" for name, dist in high_skew[:3])
                parts.append(f"以下变量呈现明显偏态分布：{skew_names}。")

        # 相关性
        if correlations:
            strong_corrs = [c for c in correlations if abs(c.coefficient) > 0.5]
            if strong_corrs:
                corr_descs = []
                for c in strong_corrs[:3]:
                    direction = "正" if c.coefficient > 0 else "负"
                    corr_descs.append(
                        f"「{c.var1}」与「{c.var2}」呈强{direction}相关（r={c.coefficient}）"
                    )
                parts.append(f"发现强相关性：{'；'.join(corr_descs)}。")

        # 异常值
        if anomalies:
            parts.append(f"共检测到 {len(anomalies)} 个异常值，建议进一步审查。")

        return "\n\n".join(parts)
