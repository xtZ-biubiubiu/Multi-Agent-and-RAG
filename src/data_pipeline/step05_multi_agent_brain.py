import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

#  引入配置文件
import config
from step04_test_retrieval_search_bm25_rerank import execute_legal_search

# ==========================================
# 1. 定义智能体之间的“共享白板” (State)
# ==========================================
class LegalCaseState(TypedDict):
    user_query: str
    search_keywords: str
    retrieved_evidence: str
    draft_opinion: str
    review_feedback: str
    loop_count: int
    is_compliant: str

# ==========================================
# 2. 初始化大模型 (全部读取 config.py)
# ==========================================
llm = ChatOllama(
    model=config.LLM_BASE_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.LLM_BASE_TEMP
)

llm_reviewer = ChatOllama(
    model=config.LLM_REVIEWER_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.LLM_REVIEWER_TEMP
)

# ==========================================
# 3. 编写 4 个智能体的具体工作逻辑 (Nodes)
# ==========================================
def query_expander(state: LegalCaseState):
    print(f"🕵️‍♂️ [案件分析员 - {config.LLM_BASE_MODEL}] 正在将口语转化为专业法律术语...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深的法律案件分析员。你的任务是将用户口语化的提问，转化为适合在法律数据库中检索的、包含法律术语的句子。不要解释，直接输出转化后的检索词。"),
        ("user", "{query}")
    ])
    chain = prompt | llm
    keywords = chain.invoke({"query": state["user_query"]}).content.strip()
    print(f"   -> 提炼检索词: {keywords}")
    return {"search_keywords": keywords, "loop_count": 0}

def legal_researcher(state: LegalCaseState):
    print("📚 [法务检索员 - 检索探针] 正在双塔知识库中搜寻法条证据...")
    # 动态读取检索参数
    evidence = execute_legal_search(
        state["search_keywords"],
        retrieve_k=config.RETRIEVE_K,
        rerank_k=config.RERANK_K
    )
    return {"retrieved_evidence": evidence}

def senior_lawyer(state: LegalCaseState):
    print(f"👨‍⚖️ [主审律师 - {config.LLM_BASE_MODEL}] 正在撰写法律意见书 (第 {state['loop_count'] + 1} 稿)...")

    feedback_context = f"【合规审查官的严厉打回意见】：\n{state['review_feedback']}\n请你务必修正上述问题，绝不能再犯！\n" if state.get("review_feedback") else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位顶尖的中国执业律师。请你根据【法条证据】回答用户问题。\n要求：\n1. 必须且只能引用提供的证据，绝不能捏造法条（幻觉）。\n2. 格式专业严谨，语气要能安抚客户。\n" + feedback_context),
        ("user", "【用户问题】：{query}\n\n【法条证据】：\n{evidence}")
    ])
    chain = prompt | llm
    draft = chain.invoke({"query": state["user_query"], "evidence": state["retrieved_evidence"]}).content
    return {"draft_opinion": draft, "loop_count": state["loop_count"] + 1}

def compliance_reviewer(state: LegalCaseState):
    print(f"⚖️ [合规审查官 - {config.LLM_REVIEWER_MODEL}] 正在进行极其严厉的法条防伪审查 (进入深度思考)...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是冷酷的合规审查官。仔细对比【律师草稿】和提供的【法条原文】。\n"
                   "规则：如果草稿中提到了原文中不存在的法律责任、具体数字、金额、期限，或者擅自过度承诺，立刻判定为 FAIL，并给出严厉的修改建议。\n"
                   "如果草稿完全忠实于原文且逻辑严密，判定为 PASS。\n"
                   "警告：你在提供修改建议时，严禁自己捏造任何金额或数字！所有数字必须 100% 来源于【法条原文】。\n"
                   "你的输出格式必须严格为两行（不要有任何多余的寒暄）：\n"
                   "第一行：PASS 或者 FAIL\n"
                   "第二行：具体的意见（如果是PASS则写‘合规通过’）。"),
        ("user", "【法条原文】：\n{evidence}\n\n【律师草稿】：\n{draft}")
    ])

    chain = prompt | llm_reviewer
    raw_output = chain.invoke({"evidence": state["retrieved_evidence"], "draft": state["draft_opinion"]}).content

    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    if think_match:
        print(f"   🧠 [{config.LLM_REVIEWER_MODEL} 内部思考]: {think_match.group(1).strip()[:150]}... (已折叠)")

    clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    review_lines = [line.strip() for line in clean_output.split('\n') if line.strip()]

    if not review_lines:
        print("   ⚠️ 审查官未输出有效结论，强制打回重试！")
        return {"is_compliant": "FAIL", "review_feedback": "审查官模型输出解析失败，请主审律师重写一版更严谨的。"}

    status = review_lines[0].upper()
    feedback = "\n".join(review_lines[1:]) if len(review_lines) > 1 else "合规通过"

    if "FAIL" in status:
        print(f"   ❌ 审查不通过，打回重写！原因: {feedback}")
        return {"is_compliant": "FAIL", "review_feedback": feedback}
    else:
        print("   ✅ 审查通过，准许签发！")
        return {"is_compliant": "PASS", "review_feedback": ""}

# ==========================================
# 4. 定义路由逻辑 (条件判断)
# ==========================================
def should_continue(state: LegalCaseState):
    # 读取配置中的最大循环次数
    if state["is_compliant"] == "PASS" or state["loop_count"] >= config.MAX_REWRITE_LOOPS:
        return "end"
    return "rewrite"

# ==========================================
# 5. 编排并编译 LangGraph 工作流图
# ==========================================
workflow = StateGraph(LegalCaseState)
workflow.add_node("Expander", query_expander)
workflow.add_node("Researcher", legal_researcher)
workflow.add_node("Lawyer", senior_lawyer)
workflow.add_node("Reviewer", compliance_reviewer)

workflow.add_edge("Expander", "Researcher")
workflow.add_edge("Researcher", "Lawyer")
workflow.add_edge("Lawyer", "Reviewer")

workflow.add_conditional_edges("Reviewer", should_continue, {"end": END, "rewrite": "Lawyer"})
workflow.set_entry_point("Expander")
legal_brain = workflow.compile()

# ==========================================
# 6. 测试多智能体系统
# ==========================================
if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("🏢 欢迎来到 法律咨询多智能体助手！")
    print(f"⚙️ 当前引擎: 主节点[{config.LLM_BASE_MODEL}] | 审查节点[{config.LLM_REVIEWER_MODEL}]")
    print("*" * 60 + "\n")

    test_query = "别人偷偷抄了我的包装盒设计拿去卖，我要去法院告他，最多能拿多少赔偿金？如果证据很难找怎么办？"
    final_state = legal_brain.invoke({"user_query": test_query})

    print("\n\n" + "=" * 60)
    print("📜 【最终签发的法律意见书】")
    print("=" * 60)
    print(final_state["draft_opinion"])