import os
import re
import mammoth
from tqdm import tqdm

# 自动定位项目根目录 (无论在哪里执行这个脚本，都能找对位置)
# 目录结构: Multi-Agent-and-RAG/src/data_pipeline/step01_word_to_md.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def clean_markdown_text(text: str) -> str:
    """
    数据清洗核心逻辑：去除大模型不需要的冗余噪音
    """
    if not text:
        return ""

    # 1. 把连续3个以上的空行，统一压缩为2个空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 2. 将中文全角空格替换为标准半角空格
    text = text.replace('\u3000', ' ')

    # 3. 剔除可能存在的零宽空白字符 (Word里常见脏数据)
    text = text.replace('\u200b', '')

    # 4. 确保首尾没有多余的空白字符
    return text.strip()


def run_etl_pipeline():
    print("=" * 50)
    print("🚀 [ETL Pipeline] 法律文档清洗与 Markdown 转换服务启动")
    print("=" * 50)

    # 确保输出目录存在
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 检查原始数据目录
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ 错误: 找不到原始数据目录 {RAW_DATA_DIR}")
        return

    # 扫描目录下所有的 .docx 文件
    docx_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.docx')]

    if not docx_files:
        print(f"⚠️ 未在 {RAW_DATA_DIR} 中找到 .docx 文件。")
        print("💡 提示: 如果你的文件是旧版 .doc 格式，请先用 Word 批量另存为 .docx！")
        return

    print(f"🔍 扫描到 {len(docx_files)} 份法律文档待处理...\n")

    success_count = 0
    error_list = []

    # 使用 tqdm 包装循环，生成动态进度条
    for filename in tqdm(docx_files, desc="文档转换进度", unit="份"):
        docx_path = os.path.join(RAW_DATA_DIR, filename)
        md_filename = filename.replace('.docx', '.md')
        md_path = os.path.join(PROCESSED_DATA_DIR, md_filename)

        try:
            with open(docx_path, "rb") as docx_file:
                # 核心：使用 mammoth 提取带有结构（Heading）的内容
                result = mammoth.convert_to_markdown(docx_file)
                raw_md_text = result.value

            # 执行数据清洗
            cleaned_md_text = clean_markdown_text(raw_md_text)

            # 保存为 .md 文件
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(cleaned_md_text)

            success_count += 1

        except Exception as e:
            error_list.append((filename, str(e)))

    # 打印运行报告 (面试时可以说脚本具有完善的日志和错误追踪)
    print("\n" + "=" * 50)
    print("📊 ETL 管道执行报告")
    print("=" * 50)
    print(f"✅ 成功转换: {success_count} 份")
    if error_list:
        print(f"❌ 失败文档: {len(error_list)} 份")
        for err in error_list[:5]:  # 只打印前5个错误防止刷屏
            print(f"   - {err[0]}: {err[1]}")
    print(f"\n📂 处理完毕！干净的 Markdown 文件已保存至:\n{PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    run_etl_pipeline()