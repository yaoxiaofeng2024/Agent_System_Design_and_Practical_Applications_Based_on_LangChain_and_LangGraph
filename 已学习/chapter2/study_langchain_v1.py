from langchain_openai.chat_models.base import ChatOpenAI

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from init_client import init_llm

llm: ChatOpenAI = init_llm(temperature=0.0)

# 定哟输出解释器，提取路由决策
class RouterOutputParser(BaseOutputParser):
    def parse(self, text: str)-> str:
        # 清理输出并转换为小写
        return text.strip().lower()

    def get_format_instructions(self) -> str:
        return "只输出一个词：'technical'、'billing'、'general' 或 'unknown'"

# 定义模拟子Agent处理程序
def technical_handler(query: str) ->str:
    """  技术问题处理程序 """
    print("\n--- 路由到技术支持 ---")
    return f"技术支持处理了查询：'{query}'。结果：已生成技术解决方案。"

def billing_handler(query: str) -> str:
    print("\n --- 路由到账单相关查询 ---")
    return f"账单支持处理了查询：'{query}'。结果：已提供账单信息。"

def general_handler(query:str) -> str:
    print("\n --- 路由到通用问题 ---")
    return f"一般支持处理了查询：'{query}'。结果：已提供通用解决方案。"

def unknown_handler(query:str)->str:
    print("\n --- 路由到未知问题 ---")
    return f"系统无法处理查询：'{query}'。请联系管理员。"

# 路由决策提示模版
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""分析用户查询并确定它属于哪个类别：
    - 如果查询涉及技术问题、产品功能或故障排除，返回"technical"
    - 如果查询涉及账单、付款、订阅或退款，返回"billing"
    - 如果查询是一般性问题、产品信息或公司政策，返回"general"
    - 如果查询不明确或不属于任何类别，返回"unknown"

    只返回一个词：technical, billing, general 或 unknown

    用户查询：{query}"""
)

# 创建路由决策链
router_chain = router_prompt | llm | RouterOutputParser()

# 定义路由器类
class QueryRouter:
    def __init__(self):
        self.handlers = {
            "technical": technical_handler,
            "billing": billing_handler,
            "general": general_handler,
            "unknown": unknown_handler
        }

    def router_query(self, query: str) -> str:
        """ 路由查询到适当的处理程序 """
        if not llm:
            return "语言模型未初始化，无法处理查询"
        
        # 获取路由决策
        router_decision = router_chain.invoke(input=query)

        # 根据决策选择处理程序
        handler = self.handlers.get(router_decision, self.handlers["unknown"])
        return handler(query)
 
def main():
    if not llm:
        print("llm is None")
        return

    router = QueryRouter()

    # 技术查询示例
    tech_query = "我的软件无法启动，显示错误代码0x80070005。"
    print(f"用户查询：{tech_query}")
    response = router.router_query(tech_query)
    print(f"回复：{response}")

    # 账单查询示例
    billing_query = "我如何更改我的订阅计划？"
    print(f"\n用户查询：{billing_query}")
    response = router.router_query(query=billing_query)
    print(f"回复：{response}")

    # 一般查询示例
    general_query = "你们的营业时间是什么时候？"
    print(f"\n用户查询：{general_query}")
    response = router.router_query(query=general_query)
    print(f"回复：{response}")

    # 不明确查询示例
    unclear_query = "blargh snorfle wump?"
    print(f"\n用户查询：{unclear_query}")
    response = router.router_query(query=unclear_query)
    print(f"回复：{response}")


if __name__ == "__main__":
    main()
