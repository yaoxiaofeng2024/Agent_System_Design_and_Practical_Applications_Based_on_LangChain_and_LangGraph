from typing import Any, Dict
import json
# ConversationBufferMemory：对话记忆组件，用于保存多轮对话的上下文历史
from langchain_classic.memory import ConversationBufferMemory
# BaseOutputParser：输出解析器基类，用于将大模型返回的文本转换为结构化数据
from langchain_core.output_parsers import BaseOutputParser
# PromptTemplate：提示词模板，用于构建带有变量占位符的提示词
from langchain_core.prompts import PromptTemplate

# 从本地模块初始化大模型客户端，参数 0.7 为温度值（越高输出越随机/有创意，越低越确定/保守）
from init_client import init_llm

llm = init_llm(0.7)

# ============================================================
# 自定义输出解析器：将大模型返回的文本中提取 JSON 并解析为字典
# 因为大模型可能在 JSON 前后附带说明文字，所以需要手动截取
# ============================================================
class TravelPlanParser(BaseOutputParser):
    def parse(self, text: str) -> Dict:
        try:
            # 找到第一个 '{' 和最后一个 '}' 的位置，截取中间的 JSON 字符串
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                # 将 JSON 字符串解析为 Python 字典
                return json.loads(json_str)
            else:
                # 如果模型输出中没找到 JSON 格式，就把原始文本包装成字典返回
                return {"plan": text}
        except Exception as e:
            # JSON 解析失败时的兜底处理，避免程序崩溃
            print(f"解析错误: {e}")
            return {"plan": text}

# ============================================================
# 提示词模板 1：旅行计划生成
# 花括号 {destination} 等是变量占位符，运行时会被实际值替换
# 注意：模板中用 {{ }} 表示字面量花括号（JSON的括号），单个 { } 是变量占位符
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
# 提示词模板 2：旅行计划调整
# 根据用户的新需求，基于原始计划进行修改
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
planning_prompt = PromptTemplate(
    input_variables=["destination", "duration", "budget", "interests", "travel_date"],
    template=planning_template
)

adjustment_prompt = PromptTemplate(
    input_variables=["original_plan", "adjustment_request"],
    template=adjustment_template
)

# ============================================================
# 使用 LCEL（LangChain Expression Language）语法创建处理链
# 管道符 | 表示数据流向：提示词模板 → 大模型 → 输出解析器
# 即：先填充模板 → 再调用大模型生成文本 → 最后解析为结构化数据
# ============================================================
planning_chain = planning_prompt | llm | TravelPlanParser()
adjustment_chain = adjustment_prompt | llm | TravelPlanParser()

# ConversationBufferMemory：对话缓冲记忆，保存所有历史对话记录
# 用于让 Agent "记住"之前的交互，实现多轮对话的上下文连贯
memory = ConversationBufferMemory()


# ============================================================
# 旅行规划 Agent 类：封装了"创建计划"和"调整计划"两个核心能力
# 每次操作都会将交互记录保存到 memory 中，方便后续追溯对话历史
# ============================================================
class TravelPlannerAgent:
    def __init__(self):
        self.current_plan = None  # 当前持有的旅行计划（字典格式）
        self.planning_chain = planning_chain    # 创建计划的链
        self.adjustment_chain = adjustment_chain # 调整计划的链
        self.memory = memory  # 对话记忆组件

    def create_plan(self, destination, duration, budget, interests, travel_date):
        """创建初始旅行计划：将参数填入模板，调用大模型，返回解析后的计划"""
        response: Any = self.planning_chain.invoke({
            "destination": destination,    # 目的地
            "duration": duration,          # 旅行时长
            "budget": budget,              # 预算
            "interests": interests,        # 兴趣偏好
            "travel_date": travel_date     # 出行时间
        })

        # 保存本次生成的计划，后续调整时需要基于此计划
        self.current_plan = response
        # 将本轮对话记录存入记忆（输入 + 输出）
        self.memory.save_context(
            {"input": f"创建旅行计划到{destination}，时长{duration}天，预算{budget}"},
            {"output": str(response)}
        )

        return response

    def adjust_plan(self, adjustment_request):
        """根据新的调整需求，修改当前旅行计划"""
        if not self.current_plan:
            return "没有可调整的计划，请先创建计划。"

        # 将当前计划和调整需求一起发给大模型，生成调整后的计划
        response = self.adjustment_chain.invoke({
            "original_plan": str(self.current_plan),  # 原始计划
            "adjustment_request": adjustment_request   # 用户新的调整需求
        })

        # 用调整后的计划覆盖当前计划
        self.current_plan = response
        # 同样将本轮交互存入记忆
        self.memory.save_context(
            {"input": f"调整计划：{adjustment_request}"},
            {"output": str(response)}
        )

        return response

    def get_current_plan(self):
        """获取当前持有的旅行计划"""
        return self.current_plan


# ============================================================
# 运行示例：演示 Agent 的"创建计划 → 调整计划"完整流程
# ============================================================
if __name__ == "__main__":
    # 初始化旅行规划 Agent
    agent = TravelPlannerAgent()

    # ---- 第一步：创建初始计划 ----
    print("## 创建初始旅行计划 ##")
    initial_plan = agent.create_plan(
        destination="中国北京",        # 目的地
        duration="5天",               # 旅行时长
        budget="10000元",             # 预算
        interests="传统文化、美食、古迹", # 兴趣偏好
        travel_date="2026年1月"       # 出行时间
    )

    print("初始计划:")
    # json.dumps 将字典格式化为缩进美观的 JSON 字符串，ensure_ascii=False 保留中文
    print(json.dumps(initial_plan, indent=2, ensure_ascii=False))

    # ---- 第二步：基于新需求调整计划 ----
    print("\n## 调整旅行计划 ##")
    adjusted_plan = agent.adjust_plan("预算减少到8000元，并增加一天行程")

    print("调整后的计划:")
    print(json.dumps(adjusted_plan, indent=2, ensure_ascii=False))