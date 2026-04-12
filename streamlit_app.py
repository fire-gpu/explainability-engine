"""
Explainability Engine - Streamlit Web Demo 应用

提供基于浏览器的交互式数据分析界面，支持文件上传、配置选择、
实时进度展示和报告下载等功能。
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# 将项目根目录添加到 sys.path，确保能正确导入项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.models import AnalysisConfig, ExplainabilityReport
from report.renderers.html import HTMLRenderer
from report.renderers.markdown import MarkdownRenderer

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="Explainability Engine",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义样式
# ============================================================

st.markdown("""
<style>
    /* 隐藏 Streamlit 默认的汉堡菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 侧边栏分区标题样式 */
    .sidebar-section-title {
        font-size: 14px;
        font-weight: 600;
        color: #4361ee;
        margin-top: 16px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #e0e0e0;
    }

    /* 主区域标题样式 */
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 4px;
    }

    .main-subtitle {
        font-size: 14px;
        color: #666;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 内置示例数据生成
# ============================================================


def generate_sample_data() -> pd.DataFrame:
    """生成定价策略示例数据（50 行）

    包含产品成本、定价、竞品价格、需求量、销量、利润率、
    地区和品类等字段，用于演示分析功能。

    Returns:
        pd.DataFrame: 示例数据框
    """
    np.random.seed(42)

    n = 50
    regions = ["华东", "华南", "华北", "西南", "华中"]
    categories = ["电子产品", "家居用品", "食品饮料", "服装鞋帽", "办公用品"]

    data = {
        "product_id": [f"P{i+1:03d}" for i in range(n)],
        "cost": np.round(np.random.uniform(20, 200, n), 2),
        "price": np.round(np.random.uniform(50, 500, n), 2),
        "competitor_price": np.round(np.random.uniform(40, 480, n), 2),
        "demand": np.random.randint(100, 5000, n),
        "volume": np.random.randint(50, 3000, n),
        "margin": np.round(np.random.uniform(0.05, 0.45, n), 4),
        "region": np.random.choice(regions, n),
        "category": np.random.choice(categories, n),
    }

    # 确保价格 > 成本（利润率为正）
    df = pd.DataFrame(data)
    df["price"] = df.apply(
        lambda row: max(row["price"], row["cost"] * 1.1), axis=1
    )
    df["margin"] = np.round((df["price"] - df["cost"]) / df["price"], 4)

    return df


# ============================================================
# LLM 客户端初始化
# ============================================================


def create_llm_client(api_key: str, base_url: str, model: str):
    """创建 LLM 客户端实例

    Args:
        api_key: API 密钥
        base_url: API 基础 URL
        model: 模型名称

    Returns:
        LLMClient 实例，如果配置不完整则返回 None
    """
    if not api_key:
        return None

    try:
        from llm.client import LLMClient

        kwargs = {"api_key": api_key, "model": model}
        if base_url:
            kwargs["base_url"] = base_url

        return LLMClient(**kwargs)
    except Exception as e:
        st.warning(f"LLM 客户端初始化失败: {e}")
        return None


# ============================================================
# 分析执行
# ============================================================


def run_analysis(
    file_path: str,
    config: AnalysisConfig,
    llm_client=None,
) -> ExplainabilityReport:
    """执行完整的分析流程

    创建编排器并运行分析，返回生成的报告。

    Args:
        file_path: 数据文件路径
        config: 分析配置
        llm_client: LLM 客户端实例（可选）

    Returns:
        ExplainabilityReport: 分析报告
    """
    from core.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        config=config,
        llm_client=llm_client,
    )
    return orchestrator.run(file_path)


# ============================================================
# 图表数据展示
# ============================================================


def render_charts(report: ExplainabilityReport):
    """渲染报告中的图表数据

    将图表数据转换为 Streamlit 可展示的 DataFrame 或指标卡片。

    Args:
        report: 可解释性分析报告
    """
    if not report.charts:
        st.info("当前分析未产生图表数据。")
        return

    for chart in report.charts:
        st.markdown(f"**{chart.title}**")
        if chart.description:
            st.caption(chart.description)

        if chart.type == "bar":
            # 柱状图数据展示为 DataFrame
            labels = chart.data.get("labels", [])
            values = chart.data.get("values", [])
            if labels and values:
                chart_df = pd.DataFrame({
                    "变量": labels,
                    "值": values,
                })
                st.dataframe(chart_df, use_container_width=True, hide_index=True)

        elif chart.type == "heatmap":
            # 热力图数据展示为矩阵 DataFrame
            variables = chart.data.get("variables", [])
            matrix = chart.data.get("matrix", [])
            if variables and matrix:
                chart_df = pd.DataFrame(
                    matrix,
                    index=variables,
                    columns=variables,
                )
                chart_df.index.name = "变量"
                st.dataframe(
                    chart_df.style.format("{:.4f}"),
                    use_container_width=True,
                )

        elif chart.type == "graph":
            # 因果图数据展示为边列表
            edges = chart.data.get("edges", [])
            if edges:
                chart_df = pd.DataFrame(edges)
                st.dataframe(chart_df, use_container_width=True, hide_index=True)

        else:
            # 其他类型：直接展示 data 字典
            for key, value in chart.data.items():
                st.write(f"**{key}**: {value}")

        st.divider()


# ============================================================
# 主应用
# ============================================================


def main():
    """Streamlit 主函数"""

    # ---- 标题区域 ----
    st.markdown('<div class="main-title">Explainability Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">'
        "AI 驱动的数据可解释性分析引擎 -- 将数据分析结果转化为人类可理解的叙述"
        "</div>",
        unsafe_allow_html=True,
    )

    # ================================================================
    # 侧边栏配置
    # ================================================================
    with st.sidebar:
        st.header("分析配置")

        # ---- 数据源选择 ----
        st.markdown('<div class="sidebar-section-title">数据源</div>', unsafe_allow_html=True)

        data_source = st.radio(
            label="选择数据源",
            options=["使用示例数据", "上传文件"],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        sample_df = None

        if data_source == "上传文件":
            uploaded_file = st.file_uploader(
                label="上传数据文件",
                type=["csv", "json", "xlsx", "xls"],
                help="支持 CSV、JSON、Excel 格式",
            )
        else:
            sample_df = generate_sample_data()
            st.success(f"已加载示例数据：{len(sample_df)} 行 x {len(sample_df.columns)} 列")
            with st.expander("预览示例数据"):
                st.dataframe(sample_df.head(5), use_container_width=True)

        # ---- 业务领域 ----
        st.markdown('<div class="sidebar-section-title">业务领域</div>', unsafe_allow_html=True)

        domain_map = {
            "定价策略": "pricing",
            "营销分析": "marketing",
            "风险评估": "risk",
            "通用": "generic",
        }
        domain_label = st.selectbox(
            label="选择业务领域",
            options=list(domain_map.keys()),
            index=0,
        )

        # ---- 业务问题与场景描述 ----
        st.markdown('<div class="sidebar-section-title">业务问题</div>', unsafe_allow_html=True)

        business_question = st.text_input(
            label="关键业务问题",
            placeholder="例如：为什么利润在下降？",
            key="question",
            help="描述你最想从数据中找到答案的问题",
        )

        business_context = st.text_area(
            label="业务场景描述",
            placeholder="例如：这是一个 B2B SaaS 定价分析，数据来自过去一年的订阅数据...",
            key="context",
            help="提供业务背景信息，帮助分析引擎更好地理解数据含义",
        )

        # ---- 目标变量 ----
        st.markdown('<div class="sidebar-section-title">目标变量</div>', unsafe_allow_html=True)

        # 获取可用列名
        available_columns = []
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    preview_df = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)
                elif uploaded_file.name.endswith(".json"):
                    preview_df = pd.read_json(uploaded_file, orient="records")
                    uploaded_file.seek(0)
                elif uploaded_file.name.endswith((".xlsx", ".xls")):
                    preview_df = pd.read_excel(uploaded_file)
                    uploaded_file.seek(0)
                else:
                    preview_df = pd.DataFrame()
                available_columns = list(preview_df.columns)
            except Exception:
                st.warning("无法读取文件列名，请检查文件格式。")
        elif sample_df is not None:
            available_columns = list(sample_df.columns)

        target_variable = st.selectbox(
            label="选择目标变量（可选）",
            options=["(不指定)"] + available_columns,
            index=0,
            help="指定分析的目标变量，不指定时将自动推断",
        )
        target_variable = None if target_variable == "(不指定)" else target_variable

        # ---- 受众选择 ----
        st.markdown('<div class="sidebar-section-title">受众与深度</div>', unsafe_allow_html=True)

        audience_map = {
            "业务决策者": "executive",
            "技术人员": "technical",
            "两者兼顾": "both",
        }
        audience_label = st.selectbox(
            label="目标受众",
            options=list(audience_map.keys()),
            index=2,
        )

        depth_map = {
            "摘要": "summary",
            "标准": "standard",
            "完整": "full",
        }
        depth_label = st.selectbox(
            label="分析深度",
            options=list(depth_map.keys()),
            index=1,
        )

        # ---- 开关选项 ----
        st.markdown('<div class="sidebar-section-title">分析模块</div>', unsafe_allow_html=True)

        causal_enabled = st.toggle(
            label="启用因果分析",
            value=True,
            help="开启后将执行因果推断和反事实推理",
        )

        predictive_enabled = st.toggle(
            label="启用预测模拟",
            value=True,
            help="开启后将执行场景模拟和敏感性分析",
        )

        # ---- LLM 设置（默认折叠） ----
        st.markdown('<div class="sidebar-section-title">LLM 设置（可选）</div>', unsafe_allow_html=True)

        with st.expander("配置 LLM（用于增强解释）", expanded=False):
            llm_api_key = st.text_input(
                label="API Key",
                type="password",
                placeholder="sk-...",
                help="OpenAI 兼容的 API 密钥，也可通过环境变量 OPENAI_API_KEY 设置",
            )
            llm_base_url = st.text_input(
                label="Base URL",
                placeholder="https://api.openai.com/v1",
                help="API 基础 URL，留空使用默认地址",
            )
            llm_model = st.text_input(
                label="Model",
                value="gpt-4o-mini",
                help="使用的 LLM 模型名称",
            )

        # ---- 开始分析按钮 ----
        st.divider()

        run_clicked = st.button(
            label="开始分析",
            type="primary",
            use_container_width=True,
        )

    # ================================================================
    # 主区域 - 分析执行与结果展示
    # ================================================================

    # 初始化 session_state 用于保存分析结果
    if "report" not in st.session_state:
        st.session_state.report = None
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    # ---- 点击"开始分析" ----
    if run_clicked:
        # 验证数据源
        if data_source == "上传文件" and uploaded_file is None:
            st.error("请先上传数据文件，或选择使用示例数据。")
            st.stop()

        # 构建分析配置
        config = AnalysisConfig(
            audience=audience_map[audience_label],
            depth=depth_map[depth_label],
            target_variable=target_variable,
            causal_enabled=causal_enabled,
            predictive_enabled=predictive_enabled,
            business_question=business_question or "",
            business_context=business_context or "",
        )

        # 准备数据文件路径
        temp_file_path = None
        try:
            if data_source == "上传文件" and uploaded_file is not None:
                # 将上传文件写入临时文件
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    temp_file_path = tmp.name
            elif data_source == "使用示例数据" and sample_df is not None:
                # 将示例数据保存为临时 CSV
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv"
                ) as tmp:
                    sample_df.to_csv(tmp.name, index=False)
                    temp_file_path = tmp.name

            if not temp_file_path:
                st.error("无法准备数据文件，请检查数据源配置。")
                st.stop()

            # 初始化 LLM 客户端
            llm_client = create_llm_client(llm_api_key, llm_base_url, llm_model)

            # ---- 进度展示 ----
            progress_bar = st.progress(0, text="准备中...")
            status_container = st.status("正在启动分析引擎...", expanded=True)

            # 定义分析阶段及对应进度
            stages = [
                ("解析数据文件", 20),
                ("执行描述性分析", 40),
            ]
            if causal_enabled:
                stages.append(("执行因果分析", 60))
            if predictive_enabled:
                stages.append(("执行预测模拟", 75))
            stages.append(("生成分析报告", 95))

            # 逐步更新进度
            for stage_name, progress_value in stages:
                progress_bar.progress(progress_value, text=f"{stage_name}...")
                with status_container:
                    st.write(f":hourglass_flowing_sand: {stage_name}...")
                time.sleep(0.3)  # 短暂延迟让用户看到进度变化

            # 执行分析
            with status_container:
                st.write(":rocket: 正在执行分析引擎...")

            start_time = time.time()
            report = run_analysis(
                file_path=temp_file_path,
                config=config,
                llm_client=llm_client,
            )
            elapsed = time.time() - start_time

            # 完成
            progress_bar.progress(100, text="分析完成!")
            status_container.update(
                label=f"分析完成! 耗时 {elapsed:.1f} 秒",
                state="complete",
            )

            # 保存结果到 session_state
            st.session_state.report = report
            st.session_state.analysis_done = True

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                st.warning(f"分析超时: {error_msg}")
            else:
                st.error(f"分析失败: {error_msg}")
            st.session_state.report = None
            st.session_state.analysis_done = False

        finally:
            # 清理临时文件
            if temp_file_path:
                try:
                    Path(temp_file_path).unlink(missing_ok=True)
                except Exception:
                    pass

    # ---- 展示分析结果 ----
    if st.session_state.analysis_done and st.session_state.report is not None:
        report = st.session_state.report

        # ---- 报告元信息 ----
        if report.metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="业务领域", value=report.metadata.domain or "通用")
            with col2:
                st.metric(label="数据规模", value=report.metadata.data_source or "-")
            with col3:
                st.metric(label="生成时间", value=report.metadata.generated_at[:19] if report.metadata.generated_at else "-")

            if report.metadata.analysis_config:
                cfg = report.metadata.analysis_config
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.caption(f"目标受众: {cfg.audience}")
                with col_b:
                    st.caption(f"分析深度: {cfg.depth}")
                with col_c:
                    if cfg.target_variable:
                        st.caption(f"目标变量: {cfg.target_variable}")

        st.divider()

        # ---- Tab 页展示 ----
        tab_summary, tab_detail, tab_technical, tab_charts = st.tabs([
            "执行摘要",
            "详细分析",
            "技术附录",
            "图表数据",
        ])

        with tab_summary:
            if report.executive_summary:
                st.markdown(report.executive_summary)
            else:
                st.info("执行摘要为空。")

        with tab_detail:
            if report.detailed_analysis:
                st.markdown(report.detailed_analysis)
            else:
                st.info("详细分析为空。")

        with tab_technical:
            if report.technical_appendix:
                st.markdown(report.technical_appendix)
            else:
                st.info("技术附录为空。")

        with tab_charts:
            render_charts(report)

        st.divider()

        # ---- 下载报告 ----
        st.subheader("下载报告")

        md_renderer = MarkdownRenderer()
        html_renderer = HTMLRenderer()

        md_content = md_renderer.render(report)
        html_content = html_renderer.render(report)

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                label="下载 Markdown 报告",
                data=md_content.encode("utf-8"),
                file_name="explainability_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with download_col2:
            st.download_button(
                label="下载 HTML 报告",
                data=html_content.encode("utf-8"),
                file_name="explainability_report.html",
                mime="text/html",
                use_container_width=True,
            )

    elif not st.session_state.analysis_done:
        # 初始状态提示
        st.info("请在左侧配置分析参数，然后点击「开始分析」按钮。")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    main()
