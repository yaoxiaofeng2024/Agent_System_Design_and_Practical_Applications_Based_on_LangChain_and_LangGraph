import time
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
}

request_count = 0
total_latency = 0.0


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    stream: bool = Field(default=False, description="是否使用流式输出")


class ChatResponse(BaseModel):
    question: str
    answer: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 服务启动中... 模型已加载")
    yield
    print("🛑 服务关闭中...")


app = FastAPI(
    title="客服Agent API",
    description="基于LangChain的电商客服智能问答服务",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        uptime_seconds=time.time() - start_time
    )


@app.get("/ready")
async def readiness_check():
    try:
        test_prompt = ChatPromptTemplate.from_template("回复OK")
        chain = test_prompt | llm | StrOutputParser()
        chain.invoke({})
        return {"ready": True}
    except Exception:
        raise HTTPException(status_code=503, detail="Model not ready")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global request_count, total_latency
    start = time.time()

    context = "\n\n".join([f"{k}：{v}" for k, v in knowledge_base.items()])
    prompt = ChatPromptTemplate.from_template(
        "你是一位专业的电商客服。请根据以下知识库信息回答用户问题。\n\n"
        "知识库：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "要求：只基于知识库信息回答。如果知识库中没有相关信息，请明确说明。回答："
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": request.question})

    latency = (time.time() - start) * 1000
    request_count += 1
    total_latency += latency

    return ChatResponse(
        question=request.question,
        answer=answer,
        latency_ms=round(latency, 2)
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    context = "\n\n".join([f"{k}：{v}" for k, v in knowledge_base.items()])
    prompt = ChatPromptTemplate.from_template(
        "你是一位专业的电商客服。请根据以下知识库信息回答用户问题。\n\n"
        "知识库：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "要求：只基于知识库信息回答。如果知识库中没有相关信息，请明确说明。回答："
    )
    chain = prompt | llm | StrOutputParser()

    async def event_generator():
        async for chunk in chain.astream({"context": context, "question": request.question}):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/metrics")
async def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    return {
        "total_requests": request_count,
        "average_latency_ms": round(avg_latency, 2),
        "uptime_seconds": round(time.time() - start_time, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
