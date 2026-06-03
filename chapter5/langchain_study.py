from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 1. 初始化模型
from init_client import init_llm
llm = init_llm(temperature=0.1)

# --- 2. 模拟数据 ---
# 模拟一个包含商品信息的数据库
product_database = {
        "prod_001": {
        "name": "UltraBook Pro 14",
        "brand": "FutureTech",
        "price": 12999,
        "specs": {
            "cpu": "Intel Core i7-13代",
            "ram": "16GB LPDDR5",
            "storage": "512GB NVMe SSD",
            "display": "14英寸 2.8K OLED"
        }
    },
    "prod_002": {
        "name": "MegaBook X",
        "brand": "Infinity",
        "price": 11999,
        "specs": {
            "cpu": "AMD Ryzen 7 7840HS",
            "ram": "32GB DDR5",
            "storage": "1TB NVMe SSD",
            "display": "15.6英寸 1080p IPS"
        }
    },
    "prod_003": {
        "name": "SlimAir Go",
        "brand": "CloudWalk",
        "price": 8999,
        "specs": {
            "cpu": "Apple M2",
            "ram": "8GB",
            "storage": "256GB SSD",
            "display": "13.6英寸 Liquid Retina"
        }
    }
}

# 模拟一个用户评论数据库
reviews_database = {
    "prod_001": [
        {"rating": 5, "comment": "屏幕素质惊人，性能强劲，非常满意！"},
        {"rating": 4, "comment": "一切都好，就是价格有点贵。"},
        {"rating": 5, "comment": "OLED屏幕看电影的体验无敌了。"}
    ],
    "prod_002": [
        {"rating": 5, "comment": "性价比之王，32GB内存太爽了，多任务无压力。"},
        {"rating": 3, "comment": "性能不错，但做工一般，而且有点重。"},
        {"rating": 4, "comment": "AMD处理器表现很好，续航比预想中长。"}
    ],
    "prod_003": [
        {"rating": 1, "comment": "不咋的，内存真小。"},
        {"rating": 2, "comment": "还行吧，一般办公用用。"},
        {"rating": 3, "comment": "参数一般，但苹果质量还是不错的。"}
    ]
}

# --- 3.工具定义 ---
@tool
def search_product(query: str) -> str:
    """
    根据关键词搜索商品。返回匹配商品的ID和名称。
    例如，输入 'FutureTech' 或 'UltraBook'。
    """
    print(f"\n--- 🛠️ 工具调用: search_product, 查询: '{query}' ---")
    results = []
    for pid, product in product_database.items():
        if query.lower() in product["name"].lower() or query.lower() in product["brand"].lower():
            results.append(f"ID: {pid}, 名称：{product['name']}")

    if not results:
        return f"未找到与 '{query}' 相关的商品。"

    print(f"--- 工具结果：找到{len(results)} 个商品---")
    return "\n" . join(results)

@tool
def get_product_details(product_id: str) -> str:
    """
    获取指定商品ID的详细信息，包括价格和规格。
    """
    print(f"\n--- 🛠️ 工具调用: get_product_details, 商品ID: '{product_id}' ---")
    product = product_database.get(product_id)
    if not product:
        return f"错误：未找到ID为 '{product_id}' 的商品。"

    details = (
        f"名称: {product['name']}\n"
        f"品牌: {product['brand']}\n"
        f"价格: ¥{product['price']}\n"
        f"规格:\n"
        f"  - CPU: {product['specs']['cpu']}\n"
        f"  - 内存: {product['specs']['ram']}\n"
        f"  - 存储: {product['specs']['storage']}\n"
        f"  - 屏幕: {product['specs']['display']}"
    )
    print(f"--- 工具结果: 已获取商品详情 ---")
    return details

@tool
def analyze_reviews(product_id: str) -> str:
    """
    获取并总结指定商品ID的用户评论。
    """
    print(f"\n--- 🛠️ 工具调用: analyze_reviews, 商品ID: '{product_id}' ---")
    reviews = reviews_database.get(product_id)
    if not reviews:
        return f"错误：未找到ID为 '{product_id}' 的商品评论。"

    summary = "用户评论总结:\n"
    positive_count = sum(1 for r in reviews if r['rating'] >= 4)
    negative_count = len(reviews) - positive_count

    summary += f"- 好评 ({positive_count}条): 用户普遍赞赏其性能和屏幕。\n"
    if negative_count > 0:
        summary += f"- 差评/中评 ({negative_count}条): 主要抱怨价格和重量。\n"

    summary += "\n代表性评论:\n" + "\n".join([f"- ({r['rating']}/5) {r['comment']}" for r in reviews])

    print(f"--- 工具结果: 已分析评论 ---")
    return summary

# --- 4.Agent组装 ---
tools = [search_product, get_product_details, analyze_reviews]

# 创建一个专门的提示词模版，知道Agent如何扮演购物助理
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的电子商务商品研究助理。你的任务是帮助用户比较商品，提供详细的信息和洞察。你可以使用提供的工具来搜索商品、获取详情和分析评论。请以清晰、有条理的方式呈现最终结果。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),      # Agent思考和行动的记录
])

# 创建Agent执行器，这是实际运行Agent的引擎
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建Agent执行器，这是实际运行Agent的引擎
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5.执行与演示 ---
def product_research_agent():
    query = "我想比较一下 FutureTech 和 Infinity 这两个品牌的旗舰笔记本，帮我做个决定。"
    print(f"用户问题：{query}\n")

    try:
        response = agent_executor.invoke(input={"input":query})
        print("\n --- 最终助理报告 ---")
        print(response["output"])
    except Exception as e:
        print(f"Agent执行期间发生错误：{e}")

if __name__ == "__main__":
    product_research_agent()

