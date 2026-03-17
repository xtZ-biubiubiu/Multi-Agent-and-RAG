import os
from tqdm import tqdm
# 原生 HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config
# 导入在 step02 写好的层级切分函数
from step02_hierarchical_chunking import chunk_legal_document



CURRENT_DIR = config.BASE_DIR
PROJECT_ROOT = config.PROJECT_ROOT
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
CHROMA_DB_DIR = config.CHROMA_DB_DIR


def build_chroma_vector_db():
    print("=" * 60)
    print("🚀 [Vector DB Builder] 启动企业级全本地知识库构建...")
    print("⚙️  架构: 原生 BGE-M3 (HuggingFace) + ChromaDB (Vector Store)")
    print("=" * 60)

    # 1. 使用 Python 原生环境加载 BGE-M3 模型
    print("🔌 正在通过原生 PyTorch 引擎加载 BGE-M3 模型...")
    print("⏳ 注意：首次运行会自动从 HuggingFace 下载模型权重（约2GB），请耐心等待...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': config.DEVICE},  # 如果有独显，可以换成 'cuda'
            encode_kwargs={'normalize_embeddings': True}  # 开启归一化，大幅提升检索精度
        )
        print("✅ 原生 BGE-M3 向量模型加载成功！\n")
    except Exception as e:
        print(f"❌ 模型加载失败，报错信息: {e}")
        return

    # 2. 收集所有处理好的 Markdown 文件
    md_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.md')]
    if not md_files:
        print("❌ 未在 data/processed 找到 Markdown 文件，请先运行 ETL 脚本。")
        return

    # 3. 初始化 Chroma 向量数据库集合
    vector_store = Chroma(
        collection_name="legal_knowledge_base",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"📚 开始切分并写入 Chroma 数据库，共需处理 {len(md_files)} 份法律文档：")
    total_chunks_added = 0

    # 4. 循环处理每一个文件
    for filename in tqdm(md_files, desc="文档入库进度"):
        file_path = os.path.join(PROCESSED_DATA_DIR, filename)

        try:
            chunks = chunk_legal_document(file_path)

            # 因为在 step02 已经做过空块过滤了，这里可以直接入库
            if chunks:
                vector_store.add_documents(chunks)
                total_chunks_added += len(chunks)

        except Exception as e:
            print(f"\n❌ 处理 {filename} 时发生错误: {e}")

    print("\n" + "=" * 60)
    print("🎉 Chroma 知识库构建完成！")
    print(f"📊 总计处理文档: {len(md_files)} 份")
    print(f"🧩 成功存入向量块: {total_chunks_added} 个")
    print(f"💾 数据库已持久化保存至: {CHROMA_DB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    build_chroma_vector_db()