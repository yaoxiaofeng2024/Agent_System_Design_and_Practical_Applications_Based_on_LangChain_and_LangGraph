from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from init_client import init_llm

llm = init_llm(0.3)


class MultimodalState(TypedDict):
    image_path: str
    user_question: str
    image_description: str
    analysis_result: str
    final_report: str
    base64_image: str


def image_understanding_node(state: MultimodalState):
    print("--- 节点1: 图像理解 ---")

    prompt = ChatPromptTemplate.from_template(
        "你是一位专业的图像分析师。请详细描述以下商品图片中的内容，" +
        "包括：商品类型、外观特征、颜色、材质、设计风格、适用场景等。\n\n" +
        "用户关注点：{question}\n\n" +
        "请提供详细的图像描述："
    )
    chain = prompt | llm | StrOutputParser()
    description = chain.invoke({"question": state["user_question"]})
    print(f"  图像描述: {description[:100]}...")
    return {"image_description": description}


def deep_analysis_node(state: MultimodalState):
    print("--- 节点2: 深度分析 ---")

    prompt = ChatPromptTemplate.from_template(
        "你是一位资深的商品分析专家。基于以下商品图像描述，请进行深度分析：\n\n" +
        "图像描述：{description}\n\n" +
        "用户问题：{question}\n\n" +
        "请从以下维度分析：\n" +
        "1. 商品定位：目标用户群体和市场定位\n" +
        "2. 竞争优势：相比同类产品的差异化优势\n" +
        "3. 品质评估：从外观判断品质等级\n" +
        "4. 改进建议：可能的优化方向"
    )
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({
        "description": state["image_description"],
        "question": state["user_question"]
    })
    print(f"  分析结果: {analysis[:100]}...")
    return {"analysis_result": analysis}


def report_generation_node(state: MultimodalState):
    print("--- 节点3: 报告生成 ---")

    prompt = ChatPromptTemplate.from_template(
        "你是一位专业的商品报告撰写师。请根据以下分析结果，" +
        "生成一份结构清晰、专业美观的商品分析报告。\n\n" +
        "图像描述：{description}\n\n" +
        "深度分析：{analysis}\n\n" +
        "用户问题：{question}\n\n" +
        "报告格式要求：\n" +
        "## 商品概述\n## 核心特征\n## 市场定位\n## 推荐文案\n## 改进建议"
    )
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "description": state["image_description"],
        "analysis": state["analysis_result"],
        "question": state["user_question"]
    })
    print(f"  报告生成完成")
    return {"final_report": report}


workflow = StateGraph(MultimodalState)

workflow.add_node("image_understanding", image_understanding_node)
workflow.add_node("deep_analysis", deep_analysis_node)
workflow.add_node("report_generation", report_generation_node)

workflow.add_edge(START, "image_understanding")
workflow.add_edge("image_understanding", "deep_analysis")
workflow.add_edge("deep_analysis", "report_generation")
workflow.add_edge("report_generation", END)

app = workflow.compile()

try:
    print("--- 图结构 ---")
    app.get_graph().print_ascii()
    print("\n" + "=" * 20 + "\n")
except Exception as e:
    print(f"无法打印图结构: {e}")

print("=" * 60)
print("多模态Agent - LangGraph商品分析工作流")
print("=" * 60)

test_cases = [
    {"image_path": "product.jpg", "question": "这是一款什么商品？帮我分析一下卖点"},
    {"image_path": "product.jpg", "question": "请为这个商品生成电商详情页文案"},
]

for tc in test_cases:
    print(f"\n{'='*60}")
    print(f"问题: {tc['question']}")
    print(f"{'='*60}")

    initial_state = {
        "image_path": tc["image_path"],
        "user_question": tc["question"],
        "image_description": "",
        "analysis_result": "",
        "final_report": "",
        "base64_image": ""
    }
    final_state = app.invoke(initial_state)
    print(f"\n最终报告:\n{final_state['final_report']}")

print(f"\n{'='*60}")
print("提示：实际使用时，在image_understanding_node中使用多模态消息：")
print("""
messages = [
    SystemMessage(content="你是图像分析师"),
    HumanMessage(content=[
        {"type": "text", "text": "描述这个商品"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ])
]
response = llm.invoke(messages)
""")
print(f"{'='*60}")
