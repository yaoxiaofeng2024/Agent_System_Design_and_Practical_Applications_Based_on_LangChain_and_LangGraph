# ============================================================
# 本章示例：使用 LangGraph 构建旅行规划 Agent
# 与 langchain_example.py 不同，LangGraph 用"图"的方式来组织工作流，
# 支持条件分支、循环、状态持久化等更复杂的多步骤逻辑
# ============================================================
from typing import Dict, List, Optional, TypedDict, Annotated, Literal
import json
import operator
# HumanMessage：用户消息 ｜ AIMessage：AI回复 ｜ BaseMessage：消息基类
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# BaseOutputParser：输出解析器基类 ｜ JsonOutputParser：JSON 输出解析器，自动提取 JSON
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser
# PromptTemplate：提示词模板，用 {变量名} 做占位符
from langchain_core.prompts import PromptTemplate
# StateGraph：LangGraph 的核心，用于定义有状态的工作流图
# END：图的终止节点标记
from langgraph.graph import StateGraph, END
# MemorySaver：内存检查点保存器，让图在多次调用之间保持状态
from langgraph.checkpoint.memory import MemorySaver

# 从本地模块初始化大模型客户端，0.7 为温度值（越高越随机/有创意，越低越确定/保守）
from init_client import init_llm
llm = init_llm(temperature=0.7)

# ============================================================
# 自定义输出解析器：从大模型返回的文本中提取 JSON 并解析为字典
# 大模型可能在 JSON 前后附带说明文字，所以需要手动截取
# ============================================================
class TravelPlanParser(BaseOutputParser):
    def parse(self, text:str) -> Dict:
        try:
            # 找到第一个 '{' 和最后一个 '}' 的位置，截取中间的 JSON 字符串
            start_idx = text.find('{')
            end_idx = text.rfind('}')+1
            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"plan": text}
        except Exception as e:
            print(f"解析错误：{e}")
            return {"plan": text}

# ============================================================
# 用户意图的类型定义（TypedDict）
# 用 Literal 限定 intent 只能是 "create"/"adjust"/"unknown" 三种值
# 用 Optional 表示某些字段可以为 None（比如"调整"时不需要目的地等字段）
# ============================================================
class UserIntent(TypedDict):
    intent: Literal["create", "adjust", "unknown"]      # 用户意图类型
    destination: Optional[str]                          # 目的地
    duration: Optional[str]                             # 旅行时长
    budget: Optional[str]                               # 预算
    interests: Optional[str]                            # 兴趣偏好
    travel_date: Optional[str]                          # 出行时间
    adjustment_request: Optional[str]                   # 调整请求内容


# ============================================================
# 用户意图解析器：继承 JsonOutputParser，在父类解析基础上增加兜底逻辑
# 如果大模型返回的 JSON 中没有 "intent" 字段，默认设为 "unknown"
# ============================================================
class UserIntentParser(JsonOutputParser):
    def parse(self, text: str) -> UserIntent:
        try:
            result = super().parse(text)
            if "intent" not in result:
                result["intent"] = "unknown"
            return result
        except Exception as e:
            print(f"解析错误：{e}")
            return {"intent": "unknown"}

# ============================================================
# Agent 状态定义：这是 LangGraph 图中各节点之间传递的共享状态
# 每个节点函数接收 state、返回 state 的更新部分，LangGraph 自动合并
# ============================================================
class AgentState(TypedDict):
    # messages 用 Annotated + operator.add 标注，表示新消息会追加到列表而非覆盖
    messages: Annotated[List[BaseMessage], operator.add]
    current_plan: Optional[Dict]       # 当前持有的旅行计划（字典格式）
    destination: Optional[str]         # 目的地
    duration: Optional[str]            # 旅行时长
    budget: Optional[str]              # 预算
    interests: Optional[str]           # 兴趣偏好
    travel_date: Optional[str]         # 出行时间
    adjustment_request: Optional[str]  # 调整请求内容
    user_intent: Optional[str]         # 用户意图（"create"/"adjust"/"unknown"）


# ============================================================
# 提示词模板 1：用户意图解析
# 让大模型判断用户是想"创建计划"还是"调整计划"，并提取关键信息
# {{ }} 双花括号是字面量花括号（JSON的括号），单个 { } 是变量占位符
# ============================================================
intent_extraction_template = """
你是一个旅行规划助手，需要解析用户的输入，确定他们的意图并提取相关信息。

用户输入：{user_input}

请分析用户输入，确定是创建新的旅行计划还是调整现有计划，并提取相关信息。

请以JSON格式返回结果，包含以下字段：
- intent: 用户意图，只能是 "create"（创建新计划）或 "adjust"（调整现有计划）或 "unknown"（无法确定）
- destination: 目的地（如果是创建计划）
- duration: 旅行时长（如果是创建计划）
- budget: 预算（如果是创建计划）
- interests: 兴趣偏好（如果是创建计划）
- travel_date: 出行时间（如果是创建计划）
- adjustment_request: 调整请求的具体内容（如果是调整计划）

示例：
输入："我想去巴黎玩5天，预算1万元，喜欢艺术和美食，下个月出发"
输出：{{"intent": "create", "destination": "巴黎", "duration": "5天", "budget": "1万元", "interests": "艺术和美食", "travel_date": "下个月", "adjustment_request": null}}

输入："把预算减少到8000元，增加一天行程"
输出：{{"intent": "adjust", "destination": null, "duration": null, "budget": null, "interests": null, "travel_date": null, "adjustment_request": "把预算减少到8000元，增加一天行程"}}
"""

# ============================================================
# 提示词模板 2：旅行计划生成
# 根据目的地、时长、预算等信息，让大模型生成详细的旅行计划
# ============================================================
planning_template = """
你是一位专业的旅行规划师，擅长根据客户需求创建详细的旅行计划。

客户需求：
- 目的地：{destination}
- 旅行时长：{duration}
- 预算：{budget}
- 兴趣偏好：{interests}
- 出行时间：{travel_date}

请创建一个详细的旅行计划，包括：
1. 每日行程安排
2. 推荐景点和活动
3. 餐饮建议
4. 交通方案
5. 预算分配

请以JSON格式返回计划，结构如下：
{{
  "daily_itinerary": [
    {{
      "day": 1,
      "activities": ["活动1", "活动2"],
      "meals": ["早餐建议", "午餐建议", "晚餐建议"],
      "transportation": "当日交通方案"
    }}
  ],
  "budget_breakdown": {{
    "accommodation": "预算金额",
    "food": "预算金额",
    "activities": "预算金额",
    "transportation": "预算金额"
  }},
  "general_tips": ["旅行提示1", "旅行提示2"]
}}
"""

# ============================================================
# 提示词模板 3：旅行计划调整
# 将原始计划和调整需求一起发给大模型，生成调整后的计划
# ============================================================
adjustment_template = """
根据新的情况调整旅行计划：

原始计划：
{original_plan}

新情况/调整需求：
{adjustment_request}

请提供调整后的旅行计划，保持相同的JSON格式。
"""

# ============================================================
# 用 PromptTemplate 将模板字符串封装为可调用的提示词对象
# input_variables 指明模板中需要填入的变量名列表
# ============================================================
intent_extraction_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=intent_extraction_template
)

planning_prompt = PromptTemplate(
    input_variables=["destination", "duration", "budget", "interests", "travel_date"],
    template=planning_template
)

adjustment_prompt = PromptTemplate(
    input_variables=["original_plan", "adjustment_request"],
    template=adjustment_template
)

# ============================================================
# 意图解析链（LCEL 语法）：提示词模板 → 大模型 → JSON 解析器
# 管道符 | 表示数据流向：先填充模板 → 再调用大模型 → 最后解析为结构化数据
# ============================================================
intent_extraction_chain = intent_extraction_prompt | llm | UserIntentParser()

# ============================================================
# LangGraph 节点函数定义
# 每个节点函数接收当前 state，返回 state 的更新部分
# LangGraph 会自动将返回的字典合并到 state 中
# ============================================================
def extract_travel_info(state: AgentState):
    """
    节点 1：从用户消息中提取旅行信息和意图
    流程：取最新用户消息 → 调用意图解析链 → 根据意图返回不同的状态更新
    """
    # 取消息列表中的最后一条（即用户最新输入）
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content

        # 调用意图解析链，让大模型判断用户意图并提取信息
        try:
            intent_result = intent_extraction_chain.invoke(input={"user_input": content})
            intent = intent_result.get("intent", "unknown")

            if intent == "create":
                # 用户想创建新计划：提取目的地、时长等字段，清空调整请求
                return {
                    "user_intent": "create",
                    "destination": intent_result.get("destination"),
                    "duration": intent_result.get("duration"),
                    "budget": intent_result.get("budget"),
                    "interests": intent_result.get("interests"),
                    "travel_date": intent_result.get("travel_date"),
                    "adjustment_request": None
                }
            elif intent == "adjust":
                # 用户想调整计划：提取调整请求，清空创建相关字段
                return {
                    "user_intent": "adjust",
                    "adjustment_request": intent_result.get("adjustment_request"),
                    "destination": None,
                    "duration": None,
                    "budget": None,
                    "interests": None,
                    "travel_date": None
                }
            else:
                # 无法确定意图：返回提示消息（messages 会追加到列表而非覆盖）
                return {
                    "user_intent": "unknown",
                    "messages": [AIMessage(content="抱歉，我不太理解您的需求。请明确说明是创建新的旅行计划还是调整现有计划。")]
                }
        except Exception as e:
            # 意图解析链调用失败时的兜底处理
            print(f"解析用户输入时出错: {e}")
            return {
                "user_intent": "unknown",
                "messages": [AIMessage(content="抱歉，解析您的请求时遇到了问题。请再试一次。")]
            }
    return state

def create_travel_plan(state: AgentState):
    """
    节点 2：创建旅行计划
    从 state 中取出目的地等信息，调用大模型生成计划
    """
    # 用 LCEL 语法构建链：提示词模板 → 大模型 → JSON 解析器
    response = planning_prompt | llm| TravelPlanParser()
    result = response.invoke(input={
        "destination": state.get("destination", ""),    # 目的地
        "duration": state.get("duration", ""),          # 旅行时长
        "budget": state.get("budget", ""),              # 预算
        "interests": state.get("interests", ""),        # 兴趣偏好
        "travel_date": state.get("travel_date", "")     # 出行时间
    })

    # 更新 state：保存生成的计划，同时追加一条 AI 回复消息
    return {
        "current_plan": result,
        "messages": [AIMessage(content=f"已创建旅行计划：{json.dumps(result, indent=2, ensure_ascii=False)}")]
    }

def adjust_travel_plan(state: AgentState):
    """
    节点 3：调整旅行计划
    将原始计划和调整需求一起发给大模型，生成调整后的计划
    """
    # 如果当前没有计划，无法调整
    if not state.get("current_plan"):
        return {
            "messages": [AIMessage(content="没有可调整的计划，请先创建计划。")]
        }

    # 用 LCEL 语法构建链
    response = adjustment_prompt | llm | TravelPlanParser()
    result = response.invoke(input={
        "original_plan": json.dumps(obj=state["current_plan"], indent=2, ensure_ascii=False),
        "adjustment_request": state.get("adjustment_request", "")
    })

    # 更新 state：覆盖当前计划，追加 AI 回复
    return {
        "current_plan": result,
        "messages": [AIMessage(content=f"已调整旅行计划：{json.dumps(obj=result, indent=2, ensure_ascii=False)}")]
    }

def handle_unknown_intent(state: AgentState):
    """
    节点 4：处理无法识别的意图
    向用户返回提示信息，引导其重新输入
    """
    return {
        "messages": [AIMessage(content="抱歉，我不太理解您的需求。请明确说明是创建新的旅行计划还是调整现有计划。")]
    }

# ============================================================
# 路由函数：根据用户意图决定下一步走哪个节点
# 这是 add_conditional_edges 的核心，返回值为目标节点的名称字符串
# ============================================================
def route_request(state: AgentState):
    intent = state.get("user_intent", "unknown")
    if intent == "create":
        return "create_plan"
    elif intent == "adjust":
        return "adjust_plan"
    else:
        return "handle_unknown"

# ============================================================
# 构建 LangGraph 工作流图
# 图的结构：extract_info（入口）→ 条件分支 → create_plan / adjust_plan / handle_unknown → END
# ============================================================
def build_travel_planner_graph():
    """
    构建旅行规划工作流图
    图的执行流程：
    用户输入 → extract_info（提取意图）→ 条件路由 → 创建/调整/未知 → 结束
    """
    # 用 AgentState 作为图中各节点共享的状态类型
    workflow = StateGraph(AgentState)

    # ---- 添加节点 ----
    # 每个节点关联一个处理函数，节点名称字符串用于后续连线
    workflow.add_node("extract_info", extract_travel_info)
    workflow.add_node("create_plan", create_travel_plan)
    workflow.add_node("adjust_plan", adjust_travel_plan)
    workflow.add_node("handle_unknown", handle_unknown_intent)

    workflow.set_entry_point("extract_info")

    workflow.add_conditional_edges(
        source="extract_info",
        path=route_request,
        path_map={
            "create_plan": "create_plan",
            "adjust_plan": "adjust_plan",
            "handle_unknown": "handle_unknown"
        }
    )

    workflow.add_edge("create_plan", END)
    workflow.add_edge("adjust_plan", END)
    workflow.add_edge("handle_unknown", END)

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)

# ============================================================
# 旅行规划 Agent 类：封装了对 LangGraph 图的调用
# 提供三种使用方式：create_plan（结构化创建）、adjust_plan（结构化调整）、chat（自然语言对话）
# ============================================================
class TravelPlannerAgent:
    def __init__(self):
        self.graph = build_travel_planner_graph()

        # 打印图的 ASCII 结构图，方便可视化理解工作流（可选）
        try:
            print("--- 图结构 ---")
            self.graph.get_graph().print_ascii()
            print("\n" + "=" * 20 + "\n")
        except Exception as e:
            print(f"无法打印图结构: {e}")

        # 配置参数：thread_id 用于区分不同的对话会话
        # 同一个 thread_id 的多次 invoke 会共享 MemorySaver 中的状态
        self.config = {"configurable": {"thread_id": "travel_planner"}}
        self.current_plan = None  # 当前持有的旅行计划（类内部也存一份，方便直接访问）

    def create_plan(self, destination, duration, budget, interests, travel_date):
        """
        创建初始旅行计划（结构化参数方式）
        将参数组装成用户消息，传入图执行
        """
        # 组装一条结构化的用户消息
        user_message = f"创建旅行计划到{destination}，时长{duration}天，预算{budget}，兴趣偏好：{interests}，出行时间：{travel_date}"

        # 调用图执行，传入初始状态和配置
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "destination": destination,

                "duration": duration,

                "budget": budget,

                "interests": interests,

                "travel_date": travel_date
            },
            config=self.config
        )

        self.current_plan = result.get("current_plan")
        return self.current_plan
    
    def adjust_plan(self, adjustment_request):
        """
        根据新情况调整计划（结构化参数方式）
        注意：需要先调用 create_plan，否则 current_plan 为 None
        """
        user_message = f"调整计划：{adjustment_request}"

        # 调用图执行，传入当前计划和调整需求
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "adjustment_request": adjustment_request,
                "current_plan": self.current_plan  # 将当前计划也传入 state
            },
            config=self.config
        )

        self.current_plan = result.get("current_plan")
        return self.current_plan
    
    def get_current_plan(self):
        return self.current_plan

    def chat(self, user_input):
        """
        自然语言对话方式（最灵活）
        用户可以直接说"我想去北京玩5天"或"把预算减到8000"，图会自动识别意图并路由
        """
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config=self.config
        )

        self.current_plan = result.get("current_plan")
        return result["messages"][-1].content

# ============================================================
# 运行示例：演示 Agent 的三种使用方式
# 1. create_plan → 结构化创建计划
# 2. adjust_plan → 结构化调整计划
# 3. chat → 自然语言对话（最灵活，自动识别意图）
# ============================================================
if __name__ == "__main__":
    # 初始化旅行规划 Agent（会自动打印图结构）
    agent = TravelPlannerAgent()

    # ---- 方式一：结构化参数创建计划 ----
    print("## 创建初始旅行计划 ##")
    initial_plan = agent.create_plan(
        destination="中国北京",            # 目的地
        duration="5天",                   # 旅行时长
        budget="10000元",                 # 预算
        interests="传统文化、美食、古迹",   # 兴趣偏好
        travel_date="2026年1月"           # 出行时间
    )

    print("初始计划:")
    # json.dumps 将字典格式化为缩进美观的 JSON 字符串，ensure_ascii=False 保留中文
    print(json.dumps(obj=initial_plan, indent=2, ensure_ascii=False))

    # ---- 方式二：结构化参数调整计划 ----
    print("\n## 调整旅行计划 ##")
    adjusted_plan = agent.adjust_plan("预算减少到8000元，并增加一天行程")

    print("调整后的计划：")
    print(json.dumps(obj=adjusted_plan, indent=2, ensure_ascii=False))

    # ---- 方式三：自然语言对话（最灵活）----
    # 用户不需要指定是"创建"还是"调整"，大模型会自动判断意图
    print("\n## 使用聊天接口 - 自然语言输入测试 ##")

    # 测试创建计划的自然语言输入
    response1 = agent.chat(user_input="我想去美国洛杉矶玩7天，预算15000元，喜欢好莱坞和美食，明年春天出发")
    print(response1)

    response2 = agent.chat(user_input="把预算减少到12000元，增加一些购物时间")
    print(response2)
