from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, START, END

from init_client import init_llm

llm = init_llm(0.1)

knowledge_base = {
    "退货政策": "自收到商品之日起7天内可申请无理由退货。商品需保持原包装完好，不影响二次销售。" +
                "退款将在收到退货后3-5个工作日内原路返回。定制商品、内衣等特殊品类不支持无理由退货。",
    "配送时效": "标准配送：下单后2-3个工作日送达。加急配送：下单后次日送达（需额外支付15元加急费）。" +
                "偏远地区配送时间可能延长1-2天。配送范围覆盖全国（不含港澳台及部分偏远乡镇）。",
    "支付方式": "支持支付宝、微信支付、银联卡、信用卡（Visa/MasterCard）和花呗分期。" +
                "花呗分期支持3期、6期、12期，3期免息，6期和12期收取手续费。订单金额需满100元才可使用花呗分期。",
    "会员权益": "普通会员：享受9.5折优惠。银卡会员（年消费满2000元）：享受9折优惠，每月1张免邮券。" +
                "金卡会员（年消费满5000元）：享受8.5折优惠，每月2张免邮券，专属客服。" +
                "钻石会员（年消费满10000元）：享受8折优惠，每月3张免邮券，专属客服，生日礼。",
    "发票开具": "支持电子发票和纸质发票。电子发票在下单时选择，订单完成后自动发送至邮箱。" +
                "纸质发票需在下单时备注，随商品一起寄出。发票抬头默认为收货人姓名，可修改为单位名称。" +
                "增值税专用发票需提供企业资质，联系客服申请。"
}


class EvalResult(BaseModel):
    score: int = Field(description="1-5分")
    reasoning: str = Field(description="评分理由")


class EvalState(TypedDict):
    test_cases: List[dict]
    current_index: int
    results: Annotated[List[dict], "评测结果列表"]
    avg_score: float
    summary: str


def agent_answer_node(state: EvalState):
    idx = state["current_index"]
    tc = state["test_cases"][idx]
    print(f"--- 节点1: 回答测试用例 {idx+1}/{len(state['test_cases'])} ---")
    print(f"  问题: {tc['question']}")

    context = "\n\n".join([f"{k}：{v}" for k, v in knowledge_base.items()])
    prompt = ChatPromptTemplate.from_template(
        "你是一位专业的电商客服。请根据以下知识库信息回答用户问题。\n\n"
        "知识库：\n{context}\n\n"
        "用户问题：{question}\n\n要求：只基于知识库信息回答。如果知识库中没有相关信息，请明确说明。回答："
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": tc["question"]})
    print(f"  回答: {answer[:80]}...")

    tc["answer"] = answer
    tc["context"] = context
    return {"test_cases": state["test_cases"]}


def judge_node(state: EvalState):
    idx = state["current_index"]
    tc = state["test_cases"][idx]
    print(f"--- 节点2: 评估测试用例 {idx+1} ---")

    judge_prompt = ChatPromptTemplate.from_template(
        "你是一位严格的AI系统评测专家。请评估以下客服系统的回答质量。\n\n"
        "用户问题：{question}\n系统回答：{answer}\n参考上下文：{context}\n\n"
        "请从忠实度、相关性、完整性、专业性四个维度综合评估（1-5分），给出评分和理由。"
    )
    parser = JsonOutputParser(pydantic_object=EvalResult)
    chain = judge_prompt | llm | parser
    result = chain.invoke({
        "question": tc["question"],
        "answer": tc["answer"],
        "context": tc["context"]
    })
    eval_result = EvalResult(**result)
    print(f"  评分: {eval_result.score}/5, 理由: {eval_result.reasoning[:60]}...")

    new_results = state["results"] + [{
        "question": tc["question"],
        "answer": tc["answer"],
        "score": eval_result.score,
        "reasoning": eval_result.reasoning
    }]
    return {"results": new_results, "current_index": idx + 1}


def should_continue(state: EvalState) -> str:
    if state["current_index"] < len(state["test_cases"]):
        return "agent_answer"
    return "summarize"


def summarize_node(state: EvalState):
    print("--- 节点3: 汇总评测结果 ---")
    results = state["results"]
    avg_score = sum(r["score"] for r in results) / len(results)

    summary = f"共评测 {len(results)} 个用例，平均得分 {avg_score:.2f}/5。\n"
    for i, r in enumerate(results):
        summary += f"  用例{i+1}: {r['question'][:20]}... → {r['score']}/5\n"
    summary += f"评测结论: {'优秀' if avg_score >= 4 else '良好' if avg_score >= 3 else '需改进'}"

    print(summary)
    return {"avg_score": avg_score, "summary": summary}


workflow = StateGraph(EvalState)

workflow.add_node("agent_answer", agent_answer_node)
workflow.add_node("judge", judge_node)
workflow.add_node("summarize", summarize_node)

workflow.add_edge(START, "agent_answer")
workflow.add_edge("agent_answer", "judge")
workflow.add_conditional_edges("judge", should_continue, {"agent_answer": "agent_answer", "summarize": "summarize"})
workflow.add_edge("summarize", END)

app = workflow.compile()

try:
    print("--- 图结构 ---")
    app.get_graph().print_ascii()
    print("\n" + "=" * 20 + "\n")
except Exception as e:
    print(f"无法打印图结构: {e}")

test_cases = [
    {"question": "我想退货，需要满足什么条件？"},
    {"question": "加急配送多久能到？多少钱？"},
    {"question": "花呗分期有什么条件？"},
    {"question": "金卡会员有什么权益？"},
    {"question": "怎么开增值税专用发票？"},
    {"question": "你们支持货到付款吗？"},
]

initial_state = {
    "test_cases": test_cases,
    "current_index": 0,
    "results": [],
    "avg_score": 0.0,
    "summary": ""
}

print("=" * 70)
print("客服Agent评测报告 - LangGraph评测工作流")
print("=" * 70)

final_state = app.invoke(initial_state)

print(f"\n{'='*70}")
print(final_state["summary"])
print(f"{'='*70}")
