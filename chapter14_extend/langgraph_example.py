from typing import TypedDict, List, Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
import faiss

from init_client import init_llm

llm = init_llm(0.1)

tech_docs = [
    Document(
        page_content="API网关是微服务架构中的核心组件，负责请求路由、负载均衡、认证授权和限流熔断。" +
                     "常见的API网关包括Kong、Nginx、Spring Cloud Gateway等。" +
                     "Kong基于Nginx和Lua，支持插件扩展，社区活跃。" +
                     "Spring Cloud Gateway基于Spring WebFlux，与Spring生态深度集成。" +
                     "在选型时需要考虑性能要求、团队技术栈和运维能力。",
        metadata={"source": "微服务架构指南", "section": "API网关"}
    ),
    Document(
        page_content="数据库分库分表是处理海量数据的常见方案。水平分表将一张表的数据按某种规则分散到多张结构相同的表中，" +
                     "常见的分表策略包括：按范围分表（如按时间）、按哈希分表（如按用户ID取模）。" +
                     "分库分表后需要解决跨库事务、跨库查询和数据迁移等问题。" +
                     "ShardingSphere是当前最流行的分库分表中间件，支持分片路由、读写分离和分布式事务。" +
                     "MyCat是另一个选择，但社区活跃度已下降。建议新项目优先选择ShardingSphere。",
        metadata={"source": "数据库架构实践", "section": "分库分表"}
    ),
    Document(
        page_content="Redis缓存是提升系统性能的关键手段。常见的缓存模式包括：" +
                     "Cache Aside（旁路缓存）：应用先查缓存，未命中则查数据库并回写缓存，适合读多写少场景。" +
                     "Read/Write Through（读写穿透）：应用只与缓存交互，缓存负责同步数据库，逻辑更简洁。" +
                     "Write Behind（异步写入）：应用写缓存后立即返回，缓存异步批量写入数据库，性能最高但有一致性风险。" +
                     "缓存还需要关注缓存穿透（查询不存在的数据）、缓存击穿（热点Key过期）和缓存雪崩（大量Key同时过期）三大问题。" +
                     "解决方案包括布隆过滤器、互斥锁和随机过期时间等。",
        metadata={"source": "缓存架构设计", "section": "Redis缓存"}
    ),
    Document(
        page_content="消息队列在分布式系统中承担解耦、异步和削峰三大职责。" +
                     "主流消息队列对比：Kafka吞吐量极高，适合日志采集和流处理，但不支持延时消息和死信队列。" +
                     "RabbitMQ功能丰富，支持多种交换机和路由模式，适合业务消息场景，但吞吐量不如Kafka。" +
                     "RocketMQ是阿里开源的消息队列，兼顾吞吐量和功能，支持延时消息、事务消息和消息回溯。" +
                     "Pulsar是新一代消息队列，采用存储计算分离架构，支持多租户和跨地域复制。" +
                     "选型建议：日志/流处理选Kafka，业务消息选RabbitMQ，电商/金融选RocketMQ。",
        metadata={"source": "分布式消息架构", "section": "消息队列"}
    ),
    Document(
        page_content="Kubernetes（K8s）是容器编排的事实标准。核心概念包括：" +
                     "Pod：最小调度单元，包含一个或多个容器。Deployment：管理无状态应用，支持滚动更新和回滚。" +
                     "Service：为Pod提供稳定的访问入口，支持ClusterIP、NodePort和LoadBalancer三种类型。" +
                     "ConfigMap和Secret：管理配置和敏感信息，避免硬编码。" +
                     "HPA（水平Pod自动扩缩容）：根据CPU/内存或自定义指标自动调整Pod副本数。" +
                     "生产环境建议使用Helm管理应用部署，使用Prometheus+Grafana监控，使用EFK/ELK日志收集。",
        metadata={"source": "云原生运维手册", "section": "Kubernetes"}
    )
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "],
    length_function=len
)
chunks = text_splitter.split_documents(tech_docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

sample_embedding = embedding_model.embed_query("测试")
dimension = len(sample_embedding)

index = faiss.IndexFlatIP(dimension)
vectorstore = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
vectorstore.add_documents(chunks)


class RAGState(TypedDict):
    question: str
    rewritten_query: str
    documents: Annotated[List[Document], "检索到的文档"]
    context: str
    answer: str
    need_retry: bool
    retry_count: int


def rewrite_query_node(state: RAGState):
    print("--- 节点1: 查询改写 ---")
    rewrite_prompt = ChatPromptTemplate.from_template(
        "请将以下用户问题改写为更适合在技术文档中检索的查询关键词。" +
        "保持核心意图，去除口语化表达，提取关键技术术语。\n\n" +
        "用户问题：{question}\n\n检索查询："
    )
    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": state["question"]})
    print(f"  原始问题: {state['question']}")
    print(f"  改写查询: {rewritten}")
    return {"rewritten_query": rewritten}


def retrieve_node(state: RAGState):
    print("--- 节点2: 文档检索 ---")
    query = state.get("rewritten_query", state["question"])
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )
    docs = retriever.invoke(query)
    print(f"  检索到 {len(docs)} 个文档块")
    return {"documents": docs}


def format_context_node(state: RAGState):
    print("--- 节点3: 上下文格式化 ---")
    context = "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}, 章节: {doc.metadata.get('section', '未知')}]\n{doc.page_content}"
        for doc in state["documents"]
    )
    return {"context": context}


def generate_node(state: RAGState):
    print("--- 节点4: 答案生成 ---")
    rag_prompt = ChatPromptTemplate.from_template(
        "你是一位专业的技术架构师。请根据以下参考文档回答用户的问题。\n\n"
        "参考文档：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "要求：\n"
        "1. 只基于参考文档中的信息回答，不要编造\n"
        "2. 如果参考文档中没有相关信息，请明确说明\n"
        "3. 引用具体的文档来源\n"
        "回答："
    )
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    return {"answer": answer}


def evaluate_node(state: RAGState):
    print("--- 节点5: 答案评估 ---")
    if state["retry_count"] >= 1:
        print("  已达到最大重试次数，返回当前答案")
        return {"need_retry": False}

    eval_prompt = ChatPromptTemplate.from_template(
        "请评估以下RAG系统的回答质量。如果回答中包含'没有相关信息'或'无法回答'，" +
        "且这可能是因为检索不够充分，请回复'RETRY'。否则回复'PASS'。\n\n" +
        "问题：{question}\n回答：{answer}\n\n评估结果："
    )
    chain = eval_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "question": state["question"],
        "answer": state["answer"]
    }).strip()
    need_retry = "RETRY" in result
    print(f"  评估结果: {result}, 需要重试: {need_retry}")
    return {"need_retry": need_retry, "retry_count": state["retry_count"] + 1}


def should_retry(state: RAGState) -> str:
    if state["need_retry"]:
        return "rewrite_query"
    return END


workflow = StateGraph(RAGState)

workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("format_context", format_context_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)

workflow.add_edge(START, "rewrite_query")
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("retrieve", "format_context")
workflow.add_edge("format_context", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges("evaluate", should_retry, {"rewrite_query": "rewrite_query", END: END})

app = workflow.compile()

try:
    print("--- 图结构 ---")
    app.get_graph().print_ascii()
    print("\n" + "=" * 20 + "\n")
except Exception as e:
    print(f"无法打印图结构: {e}")

questions = [
    "API网关有哪些选择？怎么选型？",
    "缓存穿透、击穿和雪崩分别是什么？怎么解决？",
    "消息队列怎么选型？",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print(f"{'='*60}")
    initial_state = {
        "question": q,
        "rewritten_query": "",
        "documents": [],
        "context": "",
        "answer": "",
        "need_retry": False,
        "retry_count": 0
    }
    final_state = app.invoke(initial_state)
    print(f"\n最终回答:\n{final_state['answer']}")
