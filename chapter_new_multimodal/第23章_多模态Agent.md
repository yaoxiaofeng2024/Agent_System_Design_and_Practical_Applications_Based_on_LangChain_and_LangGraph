# 智能体实战之多模态Agent：让AI看见和理解世界

## 一.简介

前面的章节主要聚焦于文本交互的Agent，但现实世界的信息远不止文字。图像、表格、图表、截图等视觉信息在企业场景中无处不在。多模态Agent能够同时理解和处理文本与图像，是Agent从"能读"到"能看"的关键跃迁。

### 1. 什么是多模态Agent？

多模态Agent是指能够接收、理解和处理多种模态输入（文本、图像、音频等）的智能体。其核心能力包括：

* **图像理解**：识别图像中的物体、文字、图表、布局等。
* **图文推理**：结合图像内容和文本指令进行综合推理。
* **视觉问答**：针对图像内容回答用户问题。
* **文档解析**：理解包含图文混排的复杂文档。

### 2. 多模态LLM的技术演进

* **CLIP（2021）**：OpenAI提出的对比学习模型，实现图文对齐，是图文理解的基石。
* **GPT-4V（2023）**：首个强大的多模态LLM，支持图像输入和复杂视觉推理。
* **Gemini（2023）**：Google原生多模态模型，支持图像、视频、音频输入。
* **Qwen-VL（2024）**：阿里开源的多模态模型，中文理解能力强。
* **DeepSeek-VL（2024）**：DeepSeek推出的视觉语言模型，与本项目技术栈一致。

多模态LLM的工作原理：将图像通过视觉编码器（如ViT）转换为视觉Token序列，与文本Token序列拼接后，统一输入到Transformer中进行处理。

### 3. 多模态Agent的典型应用场景

* **商品图片分析**：电商场景中，用户上传商品图片，Agent自动识别商品信息、比价、推荐。
* **文档理解**：上传合同、发票、报告等文档图片，Agent提取关键信息并回答问题。
* **数据可视化解读**：上传图表截图，Agent解读数据趋势和关键指标。
* **质量检测**：制造业中，上传产品照片，Agent识别缺陷并生成检测报告。
* **医学影像辅助**：上传医学影像，Agent提供初步分析建议（辅助诊断）。

### 4. LangChain中的多模态支持

LangChain通过`HumanMessage`的内容块（Content Blocks）支持多模态输入：

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(content=[
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}}
])
```

LangGraph同样支持多模态，只需在状态中的消息列表里包含多模态消息即可。

### 5. 图像编码与传输

将图像传递给LLM有两种方式：

* **Base64编码**：将图像编码为Base64字符串，嵌入请求体。适合小图片，无需额外网络请求。
* **URL引用**：提供图像的HTTP URL，LLM服务端自行下载。适合大图片或已有URL的场景。

Base64编码示例：
```python
import base64

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
```

---

## 二.案例：商品图片分析Agent

构建一个商品图片分析Agent，用户上传商品图片，Agent自动：
1. 识别商品类型和关键特征
2. 分析商品品质和卖点
3. 生成商品描述和推荐语

场景：电商运营人员上传商品照片，Agent自动生成商品详情页文案。

---

## 三.langchain实现

### 1. 完整代码

```python
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

# --- 运行示例 ---
# 注意：需要准备一张商品图片，如 product.jpg
# 如果没有图片，可以使用模拟方式演示

print("=" * 60)
print("多模态Agent - 商品图片分析")
print("=" * 60)

# 模拟：使用文本方式演示多模态Agent的提示词设计
# 实际使用时替换为真实图片路径
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
```

---

## 四.langgraph实现

LangGraph实现多模态Agent的优势在于：可以构建包含图像理解、文本分析、结果验证等多步骤的完整工作流。

```python
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
    """图像理解节点：分析图像内容"""
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
    """深度分析节点：基于图像描述进行深度分析"""
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
    """报告生成节点：整合分析结果生成最终报告"""
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
```

---

## 五.关键知识点总结

### 多模态LLM对比

| 模型 | 图像理解 | 中文支持 | API可用性 | 适用场景 |
|------|---------|---------|----------|---------|
| GPT-4o | 极强 | 好 | OpenAI API | 通用场景 |
| Gemini Pro Vision | 极强 | 好 | Google AI | 通用场景 |
| Qwen-VL | 强 | 极好 | 阿里云/开源 | 中文场景 |
| DeepSeek-VL | 强 | 极好 | DeepSeek API | 中文场景 |
| Claude 3.5 Sonnet | 极强 | 好 | Anthropic API | 复杂推理 |

### 图像编码最佳实践

1. **压缩图像**：上传前压缩到合理大小（建议<1MB），减少Token消耗和延迟。
2. **选择格式**：JPEG适合照片，PNG适合截图和图表，WebP两者兼顾。
3. **裁剪关键区域**：如果只关心图像的某部分，先裁剪再上传。
4. **Base64 vs URL**：小图用Base64，大图或已有URL用URL方式。

### 多模态Agent设计模式

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| 单步分析 | 一次调用完成图像理解 | 简单问答 |
| 级联分析 | 先理解图像，再基于描述深度分析 | 复杂分析任务 |
| 交互式分析 | 多轮对话逐步深入 | 需要追问的场景 |
| 工具增强 | 结合OCR、目标检测等工具 | 精确提取需求 |
