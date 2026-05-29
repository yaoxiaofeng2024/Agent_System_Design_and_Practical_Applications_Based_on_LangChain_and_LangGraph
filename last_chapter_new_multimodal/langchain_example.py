import base64
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from init_client import init_llm

llm = init_llm(0.3)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_product_image(image_path: str, user_question: str = "请详细分析这个商品") -> str:
    base64_image = encode_image(image_path)

    messages = [
        SystemMessage(content="你是一位专业的电商商品分析师。你需要根据商品图片进行分析，" +
                              "包括商品类型、材质、颜色、风格、适用场景等，并生成吸引人的商品描述。"),
        HumanMessage(content=[
            {"type": "text", "text": user_question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
    ]

    response = llm.invoke(messages)
    return response.content


def generate_product_copy(image_path: str) -> dict:
    base64_image = encode_image(image_path)

    identify_prompt = HumanMessage(content=[
        {"type": "text", "text": "请识别这个商品的类型、品牌（如可见）、主要特征。以JSON格式输出。"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ])

    messages = [
        SystemMessage(content="你是商品识别专家。请识别商品信息，以JSON格式输出，包含type、brand、features字段。"),
        identify_prompt
    ]
    identify_result = llm.invoke(messages).content

    describe_prompt = HumanMessage(content=[
        {"type": "text", "text": "请为这个商品生成一段吸引人的电商详情页文案，包括：1.商品标题（20字以内）2.核心卖点（3条）3.详细描述（100字左右）4.适用人群"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ])

    messages = [
        SystemMessage(content="你是一位顶级电商文案撰写师。请根据商品图片生成吸引人的商品文案。"),
        describe_prompt
    ]
    copy_result = llm.invoke(messages).content

    return {
        "identification": identify_result,
        "copywriting": copy_result
    }


print("=" * 60)
print("多模态Agent - 商品图片分析")
print("=" * 60)

mock_analysis_prompt = ChatPromptTemplate.from_template(
    "你是一位专业的电商商品分析师。用户上传了一张商品图片，并提出了以下问题：{question}\n\n" +
    "虽然当前为模拟模式（无实际图片），请模拟分析过程，展示多模态Agent的分析思路：\n" +
    "1. 图像识别：识别商品类型、颜色、材质\n" +
    "2. 特征提取：提取关键卖点和设计亮点\n" +
    "3. 品质评估：评估商品品质和工艺\n" +
    "4. 文案生成：生成商品描述和推荐语"
)

chain = mock_analysis_prompt | llm | StrOutputParser()

questions = [
    "这是一款什么类型的商品？它的主要特征是什么？",
    "请为这个商品生成一段吸引人的电商文案",
]

for q in questions:
    print(f"\n--- 问题: {q} ---")
    result = chain.invoke({"question": q})
    print(result)

print(f"\n{'='*60}")
print("提示：实际使用时，将图片编码为Base64后通过HumanMessage的多模态content传递给LLM")
print("示例代码：")
print("""
messages = [
    SystemMessage(content="你是商品分析师"),
    HumanMessage(content=[
        {"type": "text", "text": "分析这个商品"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ])
]
response = llm.invoke(messages)
""")
print(f"{'='*60}")
