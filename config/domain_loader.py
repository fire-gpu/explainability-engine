"""
领域配置加载器模块

从 YAML 文件加载领域特定的分析配置，包括关键变量、因果假设模板、
解释重点方向和报告章节等信息。支持配置缓存和回退机制。
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.models import CausalTemplate, DomainConfig

logger = logging.getLogger(__name__)


class DomainLoader:
    """领域配置加载器

    从 config/domains/ 目录下的 YAML 文件加载领域配置。
    支持配置缓存，同一领域的配置只会被加载一次。
    当请求的领域不存在时，自动回退到 generic（通用）配置。

    使用示例::

        loader = DomainLoader()
        config = loader.load("pricing")
        print(config.name)          # 定价策略
        print(config.key_variables) # ['price', 'cost', 'demand', ...]
    """

    """领域 YAML 配置文件所在目录"""
    DOMAINS_DIR = Path(__file__).parent / "domains"

    def __init__(self) -> None:
        """初始化领域配置加载器"""
        self._cache: dict[str, DomainConfig] = {}

    def load(self, domain: str) -> DomainConfig:
        """加载指定领域的配置

        优先从缓存中获取；如果缓存未命中，则从对应的 YAML 文件加载。
        当请求的领域不存在时，自动回退到 generic（通用）配置。

        Args:
            domain: 领域标识符，如 "pricing" / "marketing" / "risk" / "generic"

        Returns:
            DomainConfig: 领域配置对象

        Raises:
            FileNotFoundError: 当请求的领域和 generic 回退配置都不存在时抛出
        """
        # 检查缓存
        if domain in self._cache:
            logger.debug("从缓存加载领域配置: %s", domain)
            return self._cache[domain]

        # 尝试加载指定领域的配置
        yaml_path = self.DOMAINS_DIR / f"{domain}.yaml"
        if yaml_path.exists():
            config = self._parse_yaml(yaml_path)
            self._cache[domain] = config
            logger.info("已加载领域配置: %s（%s）", domain, config.name)
            return config

        # 回退到 generic
        if domain != "generic":
            logger.warning(
                "领域配置文件不存在: %s，回退到 generic 配置",
                yaml_path,
            )
            return self.load("generic")

        # generic 也不存在，抛出异常
        raise FileNotFoundError(
            f"领域配置文件不存在: {yaml_path}，且 generic 回退配置也缺失"
        )

    def available_domains(self) -> list[str]:
        """返回所有可用的领域列表

        扫描 domains 目录下所有 .yaml 文件，提取领域标识符。

        Returns:
            list[str]: 可用的领域标识符列表，如 ["pricing", "marketing", "risk", "generic"]
        """
        domains: list[str] = []
        if not self.DOMAINS_DIR.exists():
            logger.warning("领域配置目录不存在: %s", self.DOMAINS_DIR)
            return domains

        for yaml_file in self.DOMAINS_DIR.glob("*.yaml"):
            # 去掉扩展名作为领域标识符
            domain_name = yaml_file.stem
            domains.append(domain_name)

        logger.debug("可用领域列表: %s", domains)
        return sorted(domains)

    def _parse_yaml(self, yaml_path: Path) -> DomainConfig:
        """解析 YAML 文件为 DomainConfig

        从 YAML 文件中读取领域配置信息，包括名称、描述、
        关键变量、因果假设模板、解释重点方向和报告章节。

        Args:
            yaml_path: YAML 文件路径

        Returns:
            DomainConfig: 解析后的领域配置对象

        Raises:
            ImportError: 当未安装 PyYAML 时抛出
            ValueError: 当 YAML 文件格式不正确时抛出
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "加载领域配置需要 PyYAML 库，请执行 pip install pyyaml"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"领域配置文件格式不正确，期望字典类型: {yaml_path}"
            )

        # 解析因果假设模板
        causal_templates: list[CausalTemplate] = []
        raw_templates = data.get("causal_templates", [])
        if isinstance(raw_templates, list):
            for tmpl in raw_templates:
                if isinstance(tmpl, dict):
                    causal_templates.append(
                        CausalTemplate(
                            from_var=tmpl.get("from", ""),
                            to_var=tmpl.get("to", ""),
                            description=tmpl.get("description", ""),
                        )
                    )

        # 构建领域配置对象
        config = DomainConfig(
            name=data.get("name", ""),
            description=data.get("description", ""),
            key_variables=data.get("key_variables", []) or [],
            causal_templates=causal_templates,
            explanation_focus=data.get("explanation_focus", []) or [],
            report_sections=data.get("report_sections", []) or [],
        )

        logger.debug(
            "解析领域配置完成: %s，关键变量 %d 个，因果模板 %d 个",
            config.name,
            len(config.key_variables),
            len(config.causal_templates),
        )

        return config
