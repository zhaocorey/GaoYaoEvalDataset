# ⚖️ GaoYaoEval - 皋陶多语言大模型评测数据集

![Status](https://img.shields.io/badge/Status-Active-success)
![Task](https://img.shields.io/badge/Task-Multilingual_Evaluation-blue)
![Language](https://img.shields.io/badge/Language-python-orange)

> **🌐 一站式多语言、多文化、多题型大模型能力评测框架**
>
> 皋陶 (GaoYao) 评测集致力于构建公平、全面的评价体系，支持**客观题**、**主观题**、**翻译题**等丰富评测场景，覆盖从基础理解到跨文化适应的完整能力维度。

---

## 🔗 核心资源 | Resources

* 📄 **技术报告 (Technical Report)**: [GaoYao_Multilingual_Benchmark_Technical_Report.pdf](./GaoYao_Multilingual_Benchmark_Technical_Report.pdf)
* 💻 **开源代码仓 (GitHub)**: [MindSpore Lab - GaoYaoEval](https://github.com/mindspore-lab/models/tree/master/research/huawei/GaoYaoEval)

---

## 📊 数据集全景图 | Dataset Overview

GaoYaoEval 包含 **10** 个核心评测子集，涵盖阅读理解、数学推理、跨文化认知等多个维度。

| ID | 评测集名称 (Dataset) | 题型 (Type) | 核心能力 (Capability) | 评测维度 (Dimension) | 状态 (Status) |
|:--:|:---------------------|:-----------:|:----------------------|:---------------------|:-------------:|
| **01** | `belebele` | 🧩 客观题 | 多语言阅读理解 | **Reading Comprehension** | ✅ Available |
| **02** | `mgsm` | 🧩 客观题 | 多语言数学推理 | **Math** | ✅ Available |
| **03** | `mmmlu` | 🧩 客观题 | 多学科知识综合 | **Reasoning** | ✅ Available |
| **04** | `superblend` | 🧩 客观题 | 混合领域综合能力 | **Cross-Culture** | 🚧 Coming Soon |
| **05** | `include` | 🧩 客观题 | 文化包容性评测 | **Knowledge** | ✅ Available |
| **06** | `culture_scope` | ⚖️ 混合题 | 单文化场景深度评测 | **Mono-Culture** | ✅ Available |
| **07** | `sage` | ⚖️ 混合题 | 跨文化理解与适应 | **Cross-Culture** | ✅ Available |
| **08** | `s_alpaca_eval` | 🖋️ 主观题 | 复杂指令遵循能力 | **Instruction Follow** | 🚧 Coming Soon |
| **09** | `s_mt_bench` | 🖋️ 主观题 | 多轮对话质量评估 | **Dialogue** | 🚧 Coming Soon |
| **10** | `flores` | 🔄 翻译题 | 高质量机器翻译 | **Translation** | ✅ Available |

### 📝 图例说明
* 🧩 **客观题**：标准选择题或填空题，便于自动化评分。
* 🖋️ **主观题**：开放式生成任务，侧重生成质量和逻辑。
* 🔄 **翻译题**：专注于多语言互译能力。
* ⚖️ **混合题**：包含客观与主观两种形式。

