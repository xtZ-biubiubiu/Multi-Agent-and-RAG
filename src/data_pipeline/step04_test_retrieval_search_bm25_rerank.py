import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import config
import jieba
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

CURRENT_DIR = config.BASE_DIR
PROJECT_ROOT = config.PROJECT_ROOT
CHROMA_DB_DIR = config.CHROMA_DB_DIR

# 屏蔽 jieba 烦人的初始化日志
jieba.setLogLevel(jieba.logging.INFO)


def jieba_preprocess(text: str):
    """BM25 的中文分词器：把长句子切成词语列表"""
    return jieba.lcut(text)


def test_hybrid_search(query: str, retrieve_k: int = 15, rerank_k: int = 3):
    print("=" * 60)
    print(f"🔍 [Hybrid Search] 启动 双路召回 + 交叉精排 满血测试")
    print(f"🗣️ 用户问题: '{query}'")
    print("=" * 60)

    # ==========================================
    # 模块 1：加载底层数据库与模型
    # ==========================================
    print("🔌 [1/3] 正在加载 Chroma 向量库 (用于语义召回)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': config.DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = Chroma(
        collection_name="legal_knowledge_base",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("📚 [2/3] 正在构建 BM25 倒排索引 (用于关键词精准召回)...")
    # 从 Chroma 数据库中一次性把所有文本提取出来，喂给 BM25
    db_data = vector_store.get(include=['documents', 'metadatas'])
    all_docs = [Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(db_data['documents'], db_data['metadatas'])]

    # 构建 BM25 检索器，并注入结巴分词
    bm25_retriever = BM25Retriever.from_documents(all_docs, preprocess_func=jieba_preprocess)
    bm25_retriever.k = retrieve_k

    print("🧠 [3/3] 正在加载交叉编码重排模型 (Reranker)...")
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=config.DEVICE)

    # ==========================================
    # 阶段一：双路召回 (Multi-way Recall)
    # ==========================================
    print(f"\n🌊 [第一阶段：双路召回] 正在兵分两路捞取数据 (各取 Top {retrieve_k})...")

    # 第 1 路：向量检索（懂意思，擅长模糊搜索）
    vector_docs = vector_store.similarity_search(query, k=retrieve_k)

    # 第 2 路：BM25 检索（死磕字眼，擅长数字和专有名词）
    bm25_docs = bm25_retriever.invoke(query)

    # 将两路人马捞回来的数据合并，并进行“去重” (以防两路搜到了同一个法条)
    unique_docs_dict = {}
    for doc in vector_docs + bm25_docs:
        unique_docs_dict[doc.page_content] = doc

    combined_docs = list(unique_docs_dict.values())
    print(f"🎣 召回完毕！向量与关键词共计捞回 {len(combined_docs)} 条不重复的候选法条。")

    # ==========================================
    # 阶段二：精排 (Rerank / Cross-Encoder)
    # ==========================================
    if not combined_docs:
        print("⚠️ 未检索到任何相关内容！")
        return

    print(f"🎯 [第二阶段：交叉精排] Reranker 开始对这 {len(combined_docs)} 条数据进行交叉审阅打分...")
    pairs = [[query, doc.page_content] for doc in combined_docs]
    rerank_scores = reranker.predict(pairs)

    scored_docs = list(zip(combined_docs, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # ==========================================
    # 输出最终结果
    # ==========================================
    print("\n" + "=" * 60)
    print(f"🏆 【精排最终榜单 Top {rerank_k}】")
    print("=" * 60)

    final_top_docs = scored_docs[:rerank_k]

    for i, (doc, score) in enumerate(final_top_docs):
        print(f"\n🥇 【排名 {i + 1}】 (综合匹配度得分: {score:.4f})")
        print(f"🏷️ [元数据]: {doc.metadata}")
        content = doc.page_content
        display_content = content if len(content) < 200 else content[:200] + "......"
        print(f"📄 [正文]: \n{display_content}")
        print("-" * 50)


# ======= 追加在 step04 文件底部的供 Agent 调用的接口 =======
def execute_legal_search(query: str, retrieve_k: int = 15, rerank_k: int = 3) -> str:
    """法务检索员 (Legal Researcher) 专属调用的内部接口"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': config.DEVICE},
                                       encode_kwargs={'normalize_embeddings': True})
    vector_store = Chroma(collection_name="legal_knowledge_base", embedding_function=embeddings,
                          persist_directory=CHROMA_DB_DIR)

    db_data = vector_store.get(include=['documents', 'metadatas'])
    all_docs = [Document(page_content=doc, metadata=meta or {}) for doc, meta in
                zip(db_data['documents'], db_data['metadatas'])]
    bm25_retriever = BM25Retriever.from_documents(all_docs, preprocess_func=jieba_preprocess)
    bm25_retriever.k = retrieve_k

    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=config.DEVICE)

    vector_docs = vector_store.similarity_search(query, k=retrieve_k)
    bm25_docs = bm25_retriever.invoke(query)

    unique_docs_dict = {doc.page_content: doc for doc in vector_docs + bm25_docs}
    combined_docs = list(unique_docs_dict.values())

    if not combined_docs: return "未检索到相关法律条文。"

    pairs = [[query, doc.page_content] for doc in combined_docs]
    rerank_scores = reranker.predict(pairs)
    scored_docs = list(zip(combined_docs, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    result_text = ""
    for i, (doc, _) in enumerate(scored_docs[:rerank_k]):
        result_text += f"【证据 {i + 1}】(元数据: {doc.metadata})\n正文: {doc.page_content}\n\n"
    return result_text

if __name__ == "__main__":
    # 测试：这是一个典型的“死磕字眼”+“特定数字”的问题，纯向量很容易翻车，BM25 能完美救场！
    test_question = "中华人民共和国专利法的第四次修正是在哪一年哪一月哪一日通过的？"
    test_hybrid_search(test_question, retrieve_k=15, rerank_k=3)