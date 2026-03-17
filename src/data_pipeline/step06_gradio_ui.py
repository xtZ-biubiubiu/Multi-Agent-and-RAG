import gradio as gr
import config

from step05_multi_agent_brain import legal_brain

# ==========================================
#  自定义 CSS 样式
# ==========================================
custom_css = """
/* 全局字体优化 */
body {
    font-family: 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', sans-serif !important;
    background-color: #f8f9fa;
}

/* 顶部 Header 专属律所风格卡片 */
.header-card {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    text-align: center;
}

.header-card h1 {
    color: white !important;
    font-size: 2.2em;
    font-weight: 600;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

.header-card p {
    color: #e2e8f0;
    font-size: 15px;
    margin: 5px 0;
}

.highlight-tag {
    background-color: rgba(255, 255, 255, 0.2);
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
    color: #ffd700;
}

/* 美化底部的折叠工作流面板 */
details.workflow-panel {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px 15px;
    margin-bottom: 20px;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
}

details.workflow-panel summary {
    font-weight: bold;
    color: #334155;
    cursor: pointer;
    font-size: 15px;
    outline: none;
}

details.workflow-panel summary:hover {
    color: #2563eb;
}

/* 最终意见书标题样式 */
.opinion-title {
    color: #1e40af;
    border-bottom: 2px solid #93c5fd;
    padding-bottom: 8px;
    margin-top: 10px;
    margin-bottom: 15px;
}
"""

def process_legal_query(message, history):
    """
    对接 Gradio 聊天框的处理函数 (Generator 流式输出版)
    """
    progress_text = "⏳ **案件受理中，AI 律所团队正在协同工作...**\n\n"
    yield progress_text

    final_opinion = ""
    total_loops = 0

    try:
        # 监听每一个节点的实时状态
        for output in legal_brain.stream({"user_query": message}):
            for node_name, state_update in output.items():

                if node_name == "Expander":
                    keywords = state_update.get("search_keywords", "")
                    progress_text += f"🕵️‍♂️ **[案件分析员]** 将口语转化为专业检索词：\n`{keywords}`\n\n"
                    yield progress_text

                elif node_name == "Researcher":
                    progress_text += f"📚 **[法务检索员]** 正在从知识库中火速捞取法条证据...\n\n"
                    yield progress_text

                elif node_name == "Lawyer":
                    loop_count = state_update.get("loop_count", 1)
                    total_loops = loop_count
                    draft = state_update.get("draft_opinion", "")
                    final_opinion = draft
                    progress_text += f"👨‍⚖️ **[主审律师]** 正在奋笔疾书，撰写第 {loop_count} 稿法律意见书...\n\n"
                    yield progress_text

                elif node_name == "Reviewer":
                    status = state_update.get("is_compliant", "")
                    feedback = state_update.get("review_feedback", "")
                    if status == "FAIL":
                        progress_text += f"❌ **[合规审查官]** 拍桌大怒，审查未通过，打回重写！\n> **驳回理由:** <span style='color:#ef4444;'>{feedback}</span>\n\n"
                    else:
                        progress_text += f"✅ **[合规审查官]** 审查通过！无逻辑漏洞，准备签发。\n\n"
                    yield progress_text

        # ==========================================
        # 流程全部跑完，组装带有高级 CSS 类的惊艳 UI
        # ==========================================
        final_ui_text = f"""<details class="workflow-panel">
<summary>⚙️ <b>点击展开查看 AI 律所内部工作流 (共经历 {total_loops} 轮审核)</b></summary>

{progress_text}
</details>

<h3 class="opinion-title">📜 最终法律意见书</h3>

{final_opinion}
"""
        yield final_ui_text

    except Exception as e:
        yield f"⚠️ 系统后台运转时发生错误，请查看运行终端的日志。\n错误详情: {str(e)}"


# ==========================================
# 🎨 构建漂亮的 Web 界面
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(title="AI 虚拟顶尖律所") as demo:

    gr.HTML(f"""
    <div class="header-card">
        <h1>⚖️ AI 虚拟顶尖律所 <span style="font-size:0.5em; vertical-align:top; opacity:0.8;">PRO版</span></h1>
        <p><b>底层算力:</b> 运行于本地 <span class="highlight-tag">RTX 4070 GPU</span></p>
        <p><b>检索引擎:</b> 法务检索员 <span class="highlight-tag">BGE-M3 语义重排引擎 + BM25 多路召回</span></p>
        <p style="margin-top: 8px;">
            <b>智能体大脑:</b> 
            案件分析/主审律师 <span class="highlight-tag">{config.LLM_BASE_MODEL}</span> 
            ⚔️ 
            合规审查官 <span class="highlight-tag">{config.LLM_REVIEWER_MODEL}</span>
        </p>
    </div>
    """)

    chat_component = gr.Chatbot(
        height=650,
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/6062/6062646.png")
    )

    gr.ChatInterface(
        fn=process_legal_query,
        chatbot=chat_component,
        examples=[
            "别人偷偷抄了我的包装盒设计拿去卖，我要去法院告他，最多能拿多少赔偿金？",
            "中华人民共和国专利法的第四次修正是在哪一年通过的？",
            "如果员工在下班路上骑电瓶车自己摔倒了，算不算工伤？"
        ],
        textbox=gr.Textbox(placeholder="请详细描述您的案件情况... (提交后可实时观看 AI 团队内部讨论)", scale=7, container=False)
    )

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 正在启动高颜值前端 Web 服务 (已开启并发队列保护)...")
    print("=" * 60 + "\n")

    # 开启队列模式
    # default_concurrency_limit=1 表示同一时间只允许 1 个任务在 GPU 上跑
    # 其他后来的用户会自动进入排队状态，并在界面上看到“等待中”和预计时间
    demo.queue(default_concurrency_limit=1)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=custom_theme,
        css=custom_css,
        share=False  # 既然你用了 Ngrok，这里设为 False 即可
    )