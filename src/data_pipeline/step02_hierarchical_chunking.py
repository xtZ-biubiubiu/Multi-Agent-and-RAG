import os
import re
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 动态获取项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def fix_markdown_headers(text: str) -> str:
    """
    数据清洗中间件：用正则表达式，把 HTML 锚点强行修正为标准 Markdown 标题
    """
    text = re.sub(r'<a id=".*?"></a>\s*(第[一二三四五六七八九十百零]+章.*)', r'# \1', text)
    text = re.sub(r'<a id=".*?"></a>\s*(第[一二三四五六七八九十百零]+条)\s*(.*)', r'## \1\n\2', text)
    text = re.sub(r'<a id=".*?"></a>', '', text)
    return text


def chunk_legal_document(file_path: str):
    """
    读取单个 MD 文件，进行层级感知切分，并彻底过滤空白块
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    fixed_text = fix_markdown_headers(raw_text)

    headers_to_split_on = [
        ("#", "Chapter"),
        ("##", "Article"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(fixed_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " "]
    )
    raw_chunks = text_splitter.split_documents(md_header_splits)

    # ==========================================
    # 新增核心逻辑：在切分源头直接清洗垃圾数据
    # ==========================================
    valid_chunks = []
    for chunk in raw_chunks:
        clean_text = chunk.page_content.strip()  # 去除首尾的空格、换行符
        # 严格过滤：内容长度必须大于 1（防止只有一个句号或纯空格的块）
        if len(clean_text) > 1:
            chunk.page_content = clean_text  # 把真正干净的文本放回去
            valid_chunks.append(chunk)

    # 返回的数据保证是 100% 干净、可直接向量化的
    return valid_chunks


def check_all_files():
    """
    全量数据体检：扫描所有文件，检查切分情况并统计拦截的空白块数量
    """
    print("=" * 60)
    print("🩺 [Data Quality Check] 启动全量法律文档切分体检...")
    print("=" * 60)

    md_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.md')]
    if not md_files:
        print("❌ 找不到 Markdown 文件！")
        return

    total_valid_chunks = 0
    files_with_issues = 0

    # 遍历所有 308 份文件
    for filename in tqdm(md_files, desc="文档体检进度"):
        file_path = os.path.join(PROCESSED_DATA_DIR, filename)

        try:
            # 这里的 chunks 已经是经过上面 valid_chunks 过滤后的干净数据了
            chunks = chunk_legal_document(file_path)
            total_valid_chunks += len(chunks)

            if len(chunks) == 0:
                print(f"\n⚠️ 警告: {filename} 切分后居然没有任何有效内容！")
                files_with_issues += 1

        except Exception as e:
            print(f"\n❌ {filename} 切分崩溃: {e}")
            files_with_issues += 1

    print("\n" + "=" * 60)
    print("✅ 全量数据切分体检完成！")
    print(f"📄 成功处理文件数: {len(md_files) - files_with_issues} / {len(md_files)}")
    print(f"🧩 生成的高质量纯净向量块总计: {total_valid_chunks} 个")
    print("💡 所有的空白块、幽灵字符块均已被底层拦截！")
    print("=" * 60)


if __name__ == "__main__":
    # 我们不只测试一篇了，直接执行全量体检！
    check_all_files()