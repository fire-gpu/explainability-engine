"""
命令行入口模块

提供 explain 命令行工具的入口点，支持完整的分析流程。
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# 将项目根目录加入 sys.path，确保模块导入正常
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--audience",
    type=click.Choice(["business", "technical", "both"]),
    default="both",
    help="目标受众类型",
)
@click.option(
    "--depth",
    type=click.Choice(["summary", "standard", "full"]),
    default="full",
    help="分析深度",
)
@click.option(
    "--domain",
    type=click.Choice(["pricing", "marketing", "risk", "generic"]),
    default="generic",
    help="业务领域",
)
@click.option(
    "--target",
    type=str,
    default=None,
    help="目标变量名",
)
@click.option(
    "--no-causal",
    is_flag=True,
    help="跳过因果分析",
)
@click.option(
    "--no-predictive",
    is_flag=True,
    help="跳过预测模拟",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="输出文件路径",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    help="输出格式",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="不使用 LLM（纯模板模式）",
)
def cli(
    file: str,
    audience: str,
    depth: str,
    domain: str,
    target: str | None,
    no_causal: bool,
    no_predictive: bool,
    output: str | None,
    output_format: str,
    no_llm: bool,
) -> None:
    """Explainability Engine - AI 模型可解释性分析引擎

    对指定数据文件执行可解释性分析，并生成自然语言报告。

    FILE: 要分析的数据文件路径（CSV、JSON、Excel 等）
    """
    # ============================================================
    # 1. 创建配置
    # ============================================================
    from core.models import AnalysisConfig

    # 将 CLI 参数映射为内部配置
    config = AnalysisConfig(
        audience=audience,
        depth=depth,
        target_variable=target,
        causal_enabled=not no_causal,
        predictive_enabled=not no_predictive,
    )

    click.echo("=" * 60)
    click.echo("Explainability Engine - AI 模型可解释性分析引擎")
    click.echo("=" * 60)
    click.echo(f"数据文件: {file}")
    click.echo(f"业务领域: {domain}")
    click.echo(f"目标受众: {audience}")
    click.echo(f"分析深度: {depth}")
    if target:
        click.echo(f"目标变量: {target}")
    click.echo(f"因果分析: {'启用' if config.causal_enabled else '跳过'}")
    click.echo(f"预测模拟: {'启用' if config.predictive_enabled else '跳过'}")
    click.echo(f"输出格式: {output_format}")
    click.echo(f"LLM 模式: {'禁用（纯模板）' if no_llm else '启用'}")
    click.echo("-" * 60)

    # ============================================================
    # 2. 创建 LLM 客户端（如果未禁用）
    # ============================================================
    llm_client = None
    if not no_llm:
        try:
            from config.settings import Settings

            settings = Settings()
            if settings.llm_api_key:
                from llm.client import LLMClient

                llm_client = LLMClient(
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url or None,
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                )
                click.echo("LLM 客户端已初始化")
            else:
                click.echo("未配置 LLM API Key，使用纯模板模式")
        except Exception as exc:
            click.echo(f"LLM 客户端初始化失败: {exc}，使用纯模板模式")

    # ============================================================
    # 3. 创建编排器
    # ============================================================
    from core.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        config=config,
        llm_client=llm_client,
    )

    # ============================================================
    # 4. 运行分析
    # ============================================================
    click.echo("\n[1/4] 开始分析...")
    try:
        report = orchestrator.run(file)
    except Exception as exc:
        click.echo(f"\n分析失败: {exc}", err=True)
        sys.exit(1)

    click.echo("[2/4] 分析完成")
    click.echo(f"[3/4] 生成报告（{output_format} 格式）...")

    # ============================================================
    # 5. 渲染报告
    # ============================================================
    if output_format == "markdown":
        from report.renderers.markdown import MarkdownRenderer

        renderer = MarkdownRenderer()
        rendered = renderer.render(report)
    elif output_format == "html":
        from report.renderers.html import HTMLRenderer

        renderer = HTMLRenderer()
        rendered = renderer.render(report)
    else:
        click.echo(f"不支持的输出格式: {output_format}", err=True)
        sys.exit(1)

    # ============================================================
    # 6. 保存到文件或输出到 stdout
    # ============================================================
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        click.echo(f"[4/4] 报告已保存到: {output}")
    else:
        click.echo("\n" + "=" * 60)
        click.echo(rendered)

    click.echo("\n分析完成！")


if __name__ == "__main__":
    cli()
