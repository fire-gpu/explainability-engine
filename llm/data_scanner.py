"""
数据预扫描模块（L1 阶段）

在正式分析之前，让 LLM 先"阅读"数据，推断业务场景和关键指标，
将理解结果注入后续分析流程，提升解释的业务相关性。
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.models import (
    AnalysisConfig,
    AnalysisInput,
    BusinessUnderstanding,
    DataMetadata,
)

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger(__name__)


class DataScanner:
    """数据预扫描器 - LLM 驱动的业务语义理解

    在正式分析之前，让 LLM 先"阅读"数据，推断业务场景和关键指标，
    将理解结果注入后续分析流程，提升解释的业务相关性。

    Args:
        llm_client: LLM 客户端实例，为 None 时使用模板回退
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """初始化数据预扫描器

        Args:
            llm_client: 可选的 LLM 客户端，用于推断业务语义
        """
        self.llm = llm_client

    def scan(self, analysis_input: AnalysisInput) -> BusinessUnderstanding:
        """执行数据预扫描

        完整流程：
        1. 构建数据摘要（列名 + 前100行 + 统计摘要）
        2. 结合用户输入的业务问题和场景
        3. 调用 LLM 推断业务语义
        4. 返回 BusinessUnderstanding

        Args:
            analysis_input: 分析输入，包含数据、元信息和配置

        Returns:
            BusinessUnderstanding: LLM 对数据的业务理解结果
        """
        df = analysis_input.data
        metadata = analysis_input.metadata
        config = analysis_input.config

        logger.info("开始 L1 数据预扫描，数据形状: %s", df.shape)

        # 构建数据摘要
        data_summary = self._build_data_summary(df, metadata)

        if self.llm is not None:
            try:
                from llm.prompts import PromptTemplates

                # 构建预扫描 prompt
                prompt = PromptTemplates.scan_prompt(
                    data_summary=data_summary,
                    business_question=config.business_question,
                    business_context=config.business_context,
                )

                # 调用 LLM
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt=PromptTemplates.SCAN_SYSTEM,
                )

                # 解析 LLM 响应
                understanding = self._parse_llm_response(response)
                logger.info("LLM 数据预扫描完成，推断场景: %s", understanding.inferred_scenario[:50])
                return understanding

            except Exception as e:
                logger.warning("LLM 数据预扫描失败，回退到模板: %s", e)

        # 模板回退
        understanding = self._template_understanding(df, metadata, config)
        logger.info("模板数据预扫描完成")
        return understanding

    def _build_data_summary(self, df: pd.DataFrame, metadata: DataMetadata) -> str:
        """构建数据摘要文本

        包含：
        - 列名和类型
        - 前 100 行数据采样
        - 各数值列的基本统计（均值、范围、缺失比例）
        - 各分类列的唯一值和频率

        Args:
            df: 输入数据框
            metadata: 数据集元信息

        Returns:
            str: 格式化的数据摘要文本
        """
        lines: list[str] = []

        # 基本信息
        lines.append(f"数据集基本信息: {df.shape[0]} 行, {df.shape[1]} 列")
        if metadata.domain:
            lines.append(f"业务领域: {metadata.domain}")
        if metadata.time_range:
            lines.append(f"时间范围: {metadata.time_range[0]} ~ {metadata.time_range[1]}")
        lines.append(f"整体缺失比例: {metadata.missing_ratio:.2%}")
        lines.append("")

        # 列名和类型
        lines.append("### 列信息")
        for col_meta in metadata.columns:
            lines.append(f"- {col_meta.name} ({col_meta.type}): {col_meta.description or '无描述'}")

        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            lines.append("\n### 数值列统计")
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    lines.append(f"- {col}: 无有效数据")
                    continue
                missing_ratio = df[col].isna().mean()
                lines.append(
                    f"- {col}: 均值={series.mean():.4f}, "
                    f"中位数={series.median():.4f}, "
                    f"标准差={series.std():.4f}, "
                    f"范围=[{series.min():.4f}, {series.max():.4f}], "
                    f"缺失比例={missing_ratio:.2%}"
                )

        # 分类列统计
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            lines.append("\n### 分类列统计")
            for col in categorical_cols:
                series = df[col].dropna()
                unique_count = series.nunique()
                missing_ratio = df[col].isna().mean()
                # 显示前 5 个高频值
                top_values = series.value_counts().head(5)
                top_str = ", ".join(
                    f"{val}({count})" for val, count in top_values.items()
                )
                lines.append(
                    f"- {col}: 唯一值数={unique_count}, "
                    f"缺失比例={missing_ratio:.2%}, "
                    f"高频值: {top_str}"
                )

        # 前 100 行数据采样
        lines.append("\n### 数据采样（前 100 行）")
        sample = df.head(100)
        lines.append(sample.to_string(max_rows=100, max_cols=20))

        return "\n".join(lines)

    def _build_scan_prompt(
        self, data_summary: str, config: AnalysisConfig
    ) -> str:
        """构建预扫描 prompt

        如果用户提供了 business_question 或 business_context，注入到 prompt 中。

        Args:
            data_summary: 数据摘要文本
            config: 分析配置

        Returns:
            str: 完整的用户提示词
        """
        from llm.prompts import PromptTemplates

        return PromptTemplates.scan_prompt(
            data_summary=data_summary,
            business_question=config.business_question,
            business_context=config.business_context,
        )

    def _parse_llm_response(self, response: str) -> BusinessUnderstanding:
        """解析 LLM 的结构化响应为 BusinessUnderstanding

        LLM 被要求返回 JSON 格式：
        {
            "inferred_scenario": "...",
            "key_metrics": ["...", "..."],
            "causal_hypotheses": ["...", "..."],
            "data_characteristics": "...",
            "analysis_suggestions": ["...", "..."],
            "business_question_answer": "..."
        }

        Args:
            response: LLM 返回的原始文本

        Returns:
            BusinessUnderstanding: 解析后的业务理解结果
        """
        try:
            # 尝试直接解析 JSON
            data = json.loads(response)
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON 块
            data = self._extract_json(response)
            if data is None:
                logger.warning("无法解析 LLM 响应为 JSON，使用空结果")
                return BusinessUnderstanding()

        return BusinessUnderstanding(
            inferred_scenario=str(data.get("inferred_scenario", "")),
            key_metrics=list(data.get("key_metrics", [])),
            causal_hypotheses=list(data.get("causal_hypotheses", [])),
            data_characteristics=str(data.get("data_characteristics", "")),
            analysis_suggestions=list(data.get("analysis_suggestions", [])),
            business_question_answer=str(data.get("business_question_answer", "")),
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """从文本中提取 JSON 块

        尝试匹配 ```json ... ``` 代码块或直接解析整个文本。

        Args:
            text: 包含 JSON 的文本

        Returns:
            dict | None: 解析成功返回字典，否则返回 None
        """
        # 尝试匹配 ```json ... ``` 代码块
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试匹配第一个 { ... } 块
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        brace_match = re.search(brace_pattern, text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _template_understanding(
        self,
        df: pd.DataFrame,
        metadata: DataMetadata,
        config: AnalysisConfig,
    ) -> BusinessUnderstanding:
        """模板回退（无 LLM 时）

        基于列名和统计特征做简单推断。

        Args:
            df: 输入数据框
            metadata: 数据集元信息
            config: 分析配置

        Returns:
            BusinessUnderstanding: 基于模板推断的业务理解结果
        """
        # 推断业务场景
        scenario_parts: list[str] = []
        if metadata.domain:
            scenario_parts.append(f"数据属于「{metadata.domain}」领域")
        if metadata.time_range:
            scenario_parts.append(
                f"时间跨度为 {metadata.time_range[0]} 至 {metadata.time_range[1]}"
            )
        scenario_parts.append(f"共 {df.shape[0]} 行 {df.shape[1]} 列数据")
        inferred_scenario = "，".join(scenario_parts) + "。"

        # 识别关键指标（基于数值列方差）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        key_metrics: list[str] = []
        if numeric_cols:
            variances = df[numeric_cols].var().fillna(0)
            top_variance_cols = variances.nlargest(min(5, len(numeric_cols))).index.tolist()
            key_metrics = top_variance_cols

        # 生成简单因果假设（基于列名关键词）
        causal_hypotheses: list[str] = []
        keyword_pairs = {
            "revenue": ["profit", "cost", "price"],
            "sales": ["marketing", "advertising", "promotion"],
            "price": ["demand", "quantity", "volume"],
            "cost": ["efficiency", "productivity", "quality"],
        }
        col_names_lower = [c.lower() for c in df.columns]
        for cause_keyword, effect_keywords in keyword_pairs.items():
            if cause_keyword in col_names_lower:
                for effect_keyword in effect_keywords:
                    if effect_keyword in col_names_lower:
                        cause_col = df.columns[col_names_lower.index(cause_keyword)]
                        effect_col = df.columns[col_names_lower.index(effect_keyword)]
                        causal_hypotheses.append(
                            f"「{cause_col}」可能影响「{effect_col}」"
                        )
                        break

        # 数据特征描述
        data_characteristics_parts: list[str] = []
        if metadata.missing_ratio > 0.1:
            data_characteristics_parts.append(
                f"数据存在较多缺失值（缺失比例 {metadata.missing_ratio:.1%}）"
            )
        if df.shape[0] < 100:
            data_characteristics_parts.append("数据量较小，分析结论需谨慎")
        elif df.shape[0] > 10000:
            data_characteristics_parts.append("数据量充足，统计结论较为可靠")
        data_characteristics = "；".join(data_characteristics_parts) if data_characteristics_parts else "数据整体质量良好"

        # 分析建议
        analysis_suggestions: list[str] = []
        if config.business_question:
            analysis_suggestions.append(
                f"重点关注与以下问题相关的分析：{config.business_question}"
            )
        if len(numeric_cols) >= 2:
            analysis_suggestions.append("建议进行变量间的相关性分析")
        if df.shape[0] >= 50:
            analysis_suggestions.append("数据量足够，可进行因果分析和预测模拟")

        # 如果用户提供了业务问题，给出简单回答
        business_question_answer = ""
        if config.business_question and key_metrics:
            business_question_answer = (
                f"基于数据初步观察，与您的问题「{config.business_question}」相关的关键指标包括："
                f"{', '.join(key_metrics)}。"
                f"建议进一步分析这些指标与其他变量的关系。"
            )

        return BusinessUnderstanding(
            inferred_scenario=inferred_scenario,
            key_metrics=key_metrics,
            causal_hypotheses=causal_hypotheses,
            data_characteristics=data_characteristics,
            analysis_suggestions=analysis_suggestions,
            business_question_answer=business_question_answer,
        )
