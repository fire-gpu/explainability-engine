# Explainability Engine

> AI 模型可解释性分析引擎 — 让模型决策透明可解释

## 简介

Explainability Engine 是一个模型无关的可解释性分析中间件。它接收模型输出文件（CSV/JSON/Excel），通过状态机编排的混合引擎（统计分析 + LLM 解释 + 规则库），系统化地完成描述性分析、因果推断和预测模拟，最终生成分层可解释性报告。

## 核心特性

- 🔄 **模型无关** — 不绑定特定模型类型，支持任意数据分析场景
- 📊 **三层解释能力** — 描述性（发生了什么）+ 因果性（为什么）+ 预测性（如果...会怎样）
- 🤖 **LLM 增强** — 统计分析保证准确性，LLM 生成高质量自然语言解释
- 📋 **分层报告** — 执行摘要（业务）+ 详细分析（业务+技术）+ 技术附录（数据科学家）
- ⚙️ **状态机编排** — 可审计、可复现、支持断点续跑
- 🏭 **领域配置** — 内置定价策略、营销分析、风险评估等领域模板
- 🛡️ **防幻觉机制** — 数值锚定、约束规则、后校验、引用溯源

## 快速开始

### 安装

```bash
pip install explainability-engine
```

### 基本使用

```bash
# 分析 CSV 文件（纯模板模式，无需 LLM）
explain data.csv --no-llm

# 使用 LLM 增强解释
export OPENAI_API_KEY=your-key
explain data.csv --domain pricing --audience both

# 指定目标变量和输出格式
explain data.csv --target revenue --format html -o report.html
```

### Python API

```python
from core.orchestrator import Orchestrator
from core.models import AnalysisConfig
from llm.client import LLMClient

config = AnalysisConfig(audience="both", depth="full")
llm = LLMClient()  # 需要设置 OPENAI_API_KEY
orchestrator = Orchestrator(config=config, llm_client=llm)
report = orchestrator.run("data.csv")
print(report.executive_summary)
```

## 在线演示

### 本地运行 Demo

项目提供了一个基于 Streamlit 的交互式 Web Demo，支持文件上传、配置选择、实时进度展示和报告下载。

```bash
# 安装 Demo 依赖
pip install -e ".[demo]"

# 启动 Streamlit 应用
streamlit run streamlit_app.py
```

启动后浏览器会自动打开 `http://localhost:8501`，即可在图形界面中体验完整的分析流程。

### 部署到 Streamlit Cloud

1. 将项目推送到 GitHub 仓库
2. 登录 [Streamlit Cloud](https://streamlit.io/cloud)，创建新应用
3. 关联 GitHub 仓库，设置以下配置：
   - **Repository**: 选择项目仓库
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **Python version**: `3.11`
4. 如需使用 LLM 功能，在应用的 **Secrets** 中添加 `OPENAI_API_KEY` 等环境变量
5. 点击 **Deploy** 即可完成部署

## 架构说明

Explainability Engine 采用状态机驱动的分层架构，核心组件包括：

```
cli/          → 命令行入口
core/         → 核心框架（状态机、编排器、数据模型）
analyzers/    → 分析模块（描述性分析、因果推断、预测模拟）
llm/          → LLM 集成（客户端封装、Prompt 模板、防幻觉校验）
rules/        → 规则引擎（流程控制、校验规则、解释模板）
report/       → 报告生成（Markdown/HTML 渲染器）
config/       → 领域配置（定价、营销、风险评估模板）
```

分析流程由状态机统一编排：`解析文件 → 描述性分析 → 因果推断 → 预测模拟 → 报告生成`，每一步可独立审计和复现。

详细的架构设计请参阅项目中的设计文档。

## CLI 参数说明

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `FILE` | — | 位置参数 | 必填 | 要分析的数据文件路径（CSV、JSON、Excel 等） |
| `--audience` | — | `business` / `technical` / `both` | `both` | 目标受众类型 |
| `--depth` | — | `summary` / `standard` / `full` | `full` | 分析深度 |
| `--domain` | — | `pricing` / `marketing` / `risk` / `generic` | `generic` | 业务领域 |
| `--target` | — | 字符串 | `None` | 目标变量名 |
| `--no-causal` | — | 标志 | `False` | 跳过因果分析 |
| `--no-predictive` | — | 标志 | `False` | 跳过预测模拟 |
| `--output` | `-o` | 文件路径 | `None` | 输出文件路径（不指定则输出到 stdout） |
| `--format` | — | `markdown` / `html` | `markdown` | 输出格式 |
| `--no-llm` | — | 标志 | `False` | 不使用 LLM（纯模板模式） |

## 领域配置

Explainability Engine 内置了多个业务领域的分析模板，通过 `--domain` 参数选择：

| 领域 | 参数值 | 说明 |
|------|--------|------|
| **定价策略** | `pricing` | 适用于价格优化、弹性分析、竞品定价等场景 |
| **营销分析** | `marketing` | 适用于营销活动效果分析、渠道归因、用户分群等场景 |
| **风险评估** | `risk` | 适用于信用评分、欺诈检测、合规风险等场景 |
| **通用分析** | `generic` | 适用于无特定领域约束的通用数据分析场景 |

每个领域模板包含：
- 领域专用的解释模板和术语映射
- 领域相关的校验规则和约束条件
- 针对性的分析指标和关注点

## 开发指南

### 环境要求

- Python >= 3.11
- 推荐使用虚拟环境

### 安装开发依赖

```bash
# 克隆仓库
git clone https://github.com/your-username/explainability-engine.git
cd explainability-engine

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装项目及开发依赖
pip install -e ".[dev]"
```

### 运行测试

```bash
# 运行全部测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=. --cov-report=term-missing
```

### 代码质量

```bash
# 代码格式检查
ruff check .

# 类型检查
mypy .
```

## License

[MIT](LICENSE)
