# 多智能体法律咨询智能助手 (AI Virtual Top Law Firm)

> 专为大模型初学者打造的端到端 RAG + Multi-Agent 实战项目。本项目模拟真实法律咨询服务场景，从零构建包含数据清洗、混合检索、多智能体协作及 Web 界面的完整链路，旨在为学生提供高质量、可运行的开源参考范本。
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## 📖 目录

- [简介](#-简介)
- [特性](#-特性)
- [截图/演示](#-截图演示)
- [快速开始](#-快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
- [核心模块讲解](#-核心模块讲解)
- [项目结构](#-项目结构)
- [贡献与学习](#-贡献与学习)

---

## 🌟 简介

当前许多学生想入门大模型应用开发（LLM App），但网络上充斥着大量只有“Hello World”级别的简单 Demo，缺乏工业级落地的完整参考。
本项目拒绝玩具代码，完全按照企业级开发标准设计：
- 全流程覆盖：从原始 Word 文档处理到最终 Web 界面交付。
- 硬核技术：实现了高阶 RAG（混合检索+重排）和多智能体（Self-Reflection）模式。
- 本地化部署：基于 Ollama + ChromaDB，无需昂贵的 API 费用，普通显卡即可运行。

## ✨ 特性

- **特性 1**：🏭 自动化 ETL 流水线：自动清洗 Word 文档，去除冗余噪音，转换为结构化 Markdown。
- **特性 2**：🧩 层级感知切分：针对法律文档特有的“章/条”结构，进行智能分层切片，保留上下文语义。
- **特性 3**：🔍 满血版混合检索 (Hybrid Search)：
    - 双路召回：结合 BM25 (关键词精准匹配) + Embeding模型BGE-M3 (语义模糊匹配)。
    - Rerank 精排：引入 Cross-Encoder 重排模型，对召回结果进行二次打分，大幅提升准确率。
- **特性 4**：多智能体协作 (Multi-Agent)：
  - 案件分析员：将口语转化为专业法律术语。
  - 法务检索员：执行高精度检索。
  - 主审律师：撰写法律文书。
  - 合规审查官：基于深度思考模型进行“找茬”，不合规直接打回重写（Self-Reflection）。
- **特性 5**：实时监控：LangSmith实时监控大模型流程。
- **特性 6**：沉浸式 Web 界面：基于 Gradio 构建，实时展示智能体内部思考与工作流状态。

## 📸 截图/演示

<img width="1494" height="916" alt="image" src="https://github.com/user-attachments/assets/8c10d082-0182-4728-b115-7bce9f40b7be" />


## 🚀 快速开始

### 环境要求

在开始之前，请确保你的开发环境满足以下要求：
- [Python](https://www.python.org/) v3.11+
- GPU: 推荐 NVIDIA 显卡 (8GB+ 显存)，或纯 CPU 模式（速度较慢）
- Ollama: 已安装并运行 qwen2.5:7b 和 deepseek-r1:8b 模型

### 安装步骤

1. 克隆本仓库：
   ```bash
   git clone https://github.com/your-username/your-project.git
   cd your-project
   
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   # 主要依赖：langchain, langgraph, chromadb, gradio, sentence-transformers, mammoth

3. 配置环境与模型：
   - 编辑 config.py，确认模型名称和本地路径。
   - 确保 Ollama 服务已启动：
   ```bash
   pip install -r requirements.txt
   # 主要依赖：langchain, langgraph, chromadb, gradio, sentence-transformers, mammoth

4. 准备数据
   - 将原始 .docx 法律文档放入 data/raw/ 目录。（我已经上传了）

5. 运行全流程
   - 按顺序执行以下步骤构建知识库并启动服务：
     ```bash
     # 1. 数据清洗 (Word -> Markdown)
     python src/data_pipeline/step01_word_to_md.py
        
     # 2. 数据切分与质检
     python src/data_pipeline/step02_hierarchical_chunking.py
        
     # 3. 构建向量数据库 (首次运行需下载模型)
     python src/vector_db/step03_build_vector_db_ChromaDB.py
        
     # 4. 测试检索效果 (可选)
     python src/retrieval/step04_test_retrieval_search_bm25_rerank.py
        
     # 5. 启动 Web 界面 (包含多智能体逻辑)
     python src/ui/step06_gradio_ui.py

   - 浏览器访问 http://localhost:7860 即可体验。

### 核心模块讲解

1. 数据处理流水线 (step01 & step02)
   不仅仅是简单的分割，更是实现了层级感知切分。
   - 痛点：传统切分会把“第 X 条”的内容切断，导致语义丢失。
   - 方案：利用 MarkdownHeaderTextSplitter 识别标题结构，配合正则清洗脏数据，确保每个 Chunk 都是完整的法律条款。
2. 高阶检索引擎 (step04)
   采用标准的 Recall + Rerank 架构。
   - 双路召回：
     - BM25：解决“特定年份”、“法条编号”等精确匹配问题。
     - Embedding (BGE-M3)：解决“语义理解”和“模糊查询”问题。
   - Rerank 精排：使用 CrossEncoder 对召回的数据进行深度相关性打分，只保留 Top 最精准的证据给 LLM，极大减少幻觉。
3. 多智能体协作大脑 (step05)
   - 基于 LangGraph 构建的状态机工作流，模拟真实律所团队：
     - 循环机制 (Loop)：引入了 Compliance Reviewer（合规审查官）。如果它发现律师草稿中有捏造的法条或数字，会返回 FAIL 并附带修改意见，强制 Senior Lawyer 重新生成，直到通过或达到最大重试次数。
     - 角色分工：
       - Expander: 优化 Prompt。
       - Researcher: 调用检索接口。
       - Lawyer: 生成内容。
       - Reviewer: 深度思考 (Deep Think) 进行质检。

### 项目结构

    ```bash
    ai-law-firm/
    ├── config.py                  # 全局配置 (模型名、路径、超参数)
    ├── data/
    │   ├── raw/                   # 原始 Word 文档
    │   ├── processed/             # 清洗后的 Markdown
    │   └── chroma_legal_db/       # 向量数据库文件
    ├── src/
    │   ├── data_pipeline/
    │   │   ├── step01_word_to_md.py       # ETL 清洗
    │   │   └── step02_hierarchical_chunking.py # 智能切分
    │   ├── vector_db/
    │   │   └── step03_build_vector_db_ChromaDB.py # 建库
    │   ├── retrieval/
    │   │   └── step04_test_retrieval_search_bm25_rerank.py # 混合检索
    │   ├── agents/
    │   │   └── step05_multi_agent_brain.py    # LangGraph 多智能体
    │   └── ui/
    │       └── step06_gradio_ui.py            # 前端界面
    ├── requirements.txt
    └── README.md


### 贡献与学习


本项目专为教育目的开源。如果你是大模型初学者，建议按以下路径学习：

- 阅读 step01 和 step02，理解为什么数据清洗比模型更重要。
- 在 step04 中尝试修改 RETRIEVE_K 和 RERANK_K 参数，观察检索结果的变化。
- 可以在 step05 中尝试增加一个新的智能体角色（例如“案例引用员”）。




Made with ❤️ for LLM Learners | 让每一个想法都能落地

祝大家拿到理想的offer！








   
