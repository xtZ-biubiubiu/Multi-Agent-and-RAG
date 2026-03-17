import os

# ==========================================
# 📂 1. 路径与环境配置 (Paths & Env)
# ==========================================
# 强制使用国内镜像，并默认开启离线模式防止断网报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_OFFLINE"] = "1"

# LangSmith 可观测性监控配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 开启全局追踪
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "填入你的LangSmith Key"  # 填入 Key
os.environ["LANGCHAIN_PROJECT"] = "AI_Lawyer_Multi_Agent"  # 项目名字


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_legal_db")

# ==========================================
# 💻 2. 硬件加速配置 (Hardware)
# ==========================================
# 开源提示：如果没有英伟达显卡，提醒他们将此处改为 "cpu"
DEVICE = "cuda"

# ==========================================
# 🧠 3. 向量与重排模型配置 (Embeddings & Reranker)
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ==========================================
# 🔍 4. 检索超参数配置 (Retrieval Hyperparameters)
# ==========================================
RETRIEVE_K = 5      # 双路召回各自捞取的粗排候选数量
RERANK_K = 2         # 经过 Cross-Encoder 精排后最终保留的极品证据数量

# ==========================================
# 🤖 5. 大语言模型与 Agent 配置 (LLM & Agent)
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434"

# 主模型：负责提炼搜索词、起草文书 (推荐指令跟随能力强的模型)
LLM_BASE_MODEL = "qwen2.5:7b"
LLM_BASE_TEMP = 0.2

# 审查模型：负责挑刺、合规检查 (推荐逻辑推理能力强的深度思考模型)
LLM_REVIEWER_MODEL = "deepseek-r1:8b"
LLM_REVIEWER_TEMP = 0.1

# Agent 循环熔断机制：最多允许打回重写几次
MAX_REWRITE_LOOPS = 2