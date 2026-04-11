"""
防幻觉校验模块

对 LLM 输出进行数值一致性、因果断言等方面的校验，
防止 LLM 生成与实际数据不符的内容。
"""

from __future__ import annotations

import re
from typing import Any


class GuardrailChecker:
    """LLM 输出防幻觉校验器

    通过检查数值一致性、因果断言合理性等方面，
    确保 LLM 生成的叙述与实际数据相符。

    Args:
        rules_engine: 规则引擎实例，用于获取校验规则
    """

    def __init__(self, rules_engine: Any) -> None:
        self.rules = rules_engine

    def check_numerical_consistency(
        self, narrative: str, numerical_data: dict[str, Any]
    ) -> list[str]:
        """检查 LLM 输出中的数值是否与实际数据一致

        从叙述文本中提取数值，并与 ``numerical_data`` 中的真实值进行对比。
        允许 5% 的相对误差（四舍五入差异）。

        Args:
            narrative: LLM 生成的叙述文本
            numerical_data: 真实数值数据，键为变量名，值为数值

        Returns:
            list[str]: 不一致的警告列表，空列表表示全部一致
        """
        warnings: list[str] = []

        # 从叙述中提取所有数值（整数和浮点数）
        numbers_in_text = re.findall(r"[-+]?\d*\.?\d+(?:,\d{3})*(?:\.\d+)?", narrative)

        # 将提取到的数值转为浮点数列表
        extracted_values: list[float] = []
        for num_str in numbers_in_text:
            # 去掉千分位逗号
            cleaned = num_str.replace(",", "")
            try:
                extracted_values.append(float(cleaned))
            except ValueError:
                continue

        # 与真实数据对比
        for key, true_value in numerical_data.items():
            if not isinstance(true_value, (int, float)):
                continue

            for extracted in extracted_values:
                # 计算相对误差
                if true_value == 0:
                    # 真值为 0 时，检查绝对差值
                    if abs(extracted - true_value) > 0.1:
                        warnings.append(
                            f"数值不一致: 变量 '{key}' 的真实值为 {true_value}，"
                            f"但叙述中出现了 {extracted}"
                        )
                else:
                    relative_error = abs(extracted - true_value) / abs(true_value)
                    if relative_error > 0.05:
                        warnings.append(
                            f"数值不一致: 变量 '{key}' 的真实值为 {true_value}，"
                            f"但叙述中出现了 {extracted}（相对误差 {relative_error:.1%}）"
                        )

        return warnings

    def check_causal_claims(
        self, narrative: str, causal_effects: list[dict[str, Any]]
    ) -> list[str]:
        """检查因果断言是否有数据支撑

        检查叙述中提到的因果关系是否存在于 ``causal_effects`` 中，
        以及因果效应的置信度是否满足阈值。

        Args:
            narrative: LLM 生成的叙述文本
            causal_effects: 已验证的因果效应列表，每个元素应包含
                ``treatment``、``outcome``、``confidence`` 等字段

        Returns:
            list[str]: 警告列表，空列表表示全部通过
        """
        warnings: list[str] = []

        # 构建已验证的因果对集合
        verified_pairs: set[tuple[str, str]] = set()
        verified_effects: dict[tuple[str, str], dict[str, Any]] = {}

        for effect in causal_effects:
            treatment = effect.get("treatment", "")
            outcome = effect.get("outcome", "")
            if treatment and outcome:
                pair = (treatment, outcome)
                verified_pairs.add(pair)
                verified_effects[pair] = effect

        # 从叙述中提取可能的因果断言模式
        # 匹配 "X 导致 Y"、"X 影响 Y"、"X 引起 Y" 等模式
        causal_patterns = [
            r"(\S+)\s*(?:导致|影响|引起|造成|使得|促使|驱动)\s*(\S+)",
            r"(\S+)\s*(?:的)?(?:变化|增加|减少|提高|降低)\s*(?:导致|影响|引起|造成)\s*(\S+)",
        ]

        mentioned_pairs: set[tuple[str, str]] = set()
        for pattern in causal_patterns:
            matches = re.findall(pattern, narrative)
            for match in matches:
                mentioned_pairs.add((match[0], match[1]))

        # 检查未验证的因果断言
        for pair in mentioned_pairs:
            if pair not in verified_pairs:
                warnings.append(
                    f"未验证的因果断言: 叙述中提到 '{pair[0]}' 对 '{pair[1]}' "
                    f"有因果影响，但该因果关系未在数据中得到验证"
                )

        # 检查置信度是否满足阈值（默认 0.8）
        confidence_threshold = 0.8
        for pair, effect in verified_effects.items():
            confidence = effect.get("confidence", effect.get("p_value"))
            if confidence is not None:
                # 如果是 p_value，需要转换为置信度
                if "p_value" in effect and "confidence" not in effect:
                    if confidence > (1 - confidence_threshold):
                        warnings.append(
                            f"因果效应置信度不足: '{pair[0]}' -> '{pair[1]}' "
                            f"的 p 值为 {confidence}，未达到 {confidence_threshold} 的置信阈值"
                        )
                elif isinstance(confidence, float) and confidence < confidence_threshold:
                    warnings.append(
                        f"因果效应置信度不足: '{pair[0]}' -> '{pair[1]}' "
                        f"的置信度为 {confidence}，低于 {confidence_threshold} 的阈值"
                    )

        return warnings

    def validate_output(
        self, narrative: str, context: dict[str, Any]
    ) -> tuple[str, list[str]]:
        """综合校验 LLM 输出，返回清理后的文本和警告列表

        依次执行数值一致性检查和因果断言检查，
        汇总所有警告并返回。

        Args:
            narrative: LLM 生成的原始叙述文本
            context: 校验上下文，支持以下键：
                - ``numerical_data``: 数值数据字典（用于数值一致性检查）
                - ``causal_effects``: 因果效应列表（用于因果断言检查）

        Returns:
            tuple[str, list[str]]: (清理后的文本, 警告列表)
        """
        all_warnings: list[str] = []

        # 数值一致性检查
        numerical_data = context.get("numerical_data")
        if numerical_data:
            warnings = self.check_numerical_consistency(narrative, numerical_data)
            all_warnings.extend(warnings)

        # 因果断言检查
        causal_effects = context.get("causal_effects")
        if causal_effects:
            warnings = self.check_causal_claims(narrative, causal_effects)
            all_warnings.extend(warnings)

        # 返回原始文本（当前不做修改）和警告列表
        return narrative, all_warnings
