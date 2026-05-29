from typing import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from init_client import init_llm

llm = init_llm(0.3)


class ResearchState(TypedDict):
    topic: str
    search_results: str
    analysis: str
    report: str


def search_node(state: ResearchState):
    print("--- LangGraph: 搜索节点 ---")
    prompt = ChatPromptTemplate.from_template(
        "你是一位技术调研员。请搜索关于'{topic}'的最新信息，" +
        "包括核心技术、主要厂商、最新进展和行业趋势。"
    )
    chain = prompt | llm | StrOutputParser()
    results = chain.invoke({"topic": state["topic"]})
    return {"search_results": results}


def analyze_node(state: ResearchState):
    print("--- LangGraph: 分析节点 ---")
    prompt = ChatPromptTemplate.from_template(
        "你是一位技术分析师。请分析以下调研信息，提取关键趋势和洞察。\n\n" +
        "调研信息：{search_results}\n\n" +
        "请从技术成熟度、市场前景、竞争格局三个维度分析。"
    )
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"search_results": state["search_results"]})
    return {"analysis": analysis}


def report_node(state: ResearchState):
    print("--- LangGraph: 报告节点 ---")
    prompt = ChatPromptTemplate.from_template(
        "你是一位技术报告撰写师。请根据以下调研和分析，生成一份结构化的技术调研报告。\n\n" +
        "调研信息：{search_results}\n\n" +
        "分析结论：{analysis}\n\n" +
        "报告格式：\n## 技术概述\n## 核心趋势\n## 市场分析\n## 建议与展望"
    )
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "search_results": state["search_results"],
        "analysis": state["analysis"]
    })
    return {"report": report}


workflow = StateGraph(ResearchState)
workflow.add_node("search", search_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("report", report_node)
workflow.add_edge(START, "search")
workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

app = workflow.compile()

try:
    print("--- 图结构 ---")
    app.get_graph().print_ascii()
    print("\n" + "=" * 20 + "\n")
except Exception as e:
    print(f"无法打印图结构: {e}")

topic = "大语言模型Agent技术"
print(f"=== LangGraph 实现（与其他框架对比基准） ===")
print(f"调研主题: {topic}")

initial_state = {"topic": topic, "search_results": "", "analysis": "", "report": ""}
final_state = app.invoke(initial_state)
print(f"\n最终报告:\n{final_state['report']}")

print(f"\n{'='*60}")
print("其他框架等价实现请参考 第24章_框架生态.md 中的代码示例")
print("CrewAI: pip install crewai，通过Agent+Task+Crew定义")
print("AutoGen: pip install pyautogen，通过ConversableAgent+GroupChat定义")
print("Google ADK: pip install google-adk，通过Agent+Runner声明式定义")
print(f"{'='*60}")
