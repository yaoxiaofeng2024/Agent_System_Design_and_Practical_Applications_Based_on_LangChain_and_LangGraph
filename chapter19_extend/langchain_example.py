import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

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
    "发票开具": "支持电子发票和纸质发票。电子发票在下单时选择，订单完成后自动发送至邮箱。" +
                "纸质发票需在下单时备注，随商品一起寄出。发票抬头默认为收货人姓名，可修改为单位名称。" +
                "增值税专用发票需提供企业资质，联系客服申请。"
}


def customer_service_agent(question: str) -> dict:
    context = "\n\n".join([f"{k}：{v}" for k, v in knowledge_base.items()])
    prompt = ChatPromptTemplate.from_template(
        "你是一位专业的电商客服。请根据以下知识库信息回答用户问题。\n\n"
        "知识库：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "要求：\n"
        "1. 只基于知识库信息回答\n"
        "2. 如果知识库中没有相关信息，请明确说明\n"
        "3. 语气友好专业\n"
        "回答："
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "context": context}


class JudgeScore(BaseModel):
    score: int = Field(description="1-5分，5分最好")
    reasoning: str = Field(description="评分理由")


def llm_as_judge(question: str, answer: str, context: str) -> JudgeScore:
    judge_prompt = ChatPromptTemplate.from_template(
        "你是一位严格的AI系统评测专家。请评估以下客服系统的回答质量。\n\n"
        "用户问题：{question}\n"
        "系统回答：{answer}\n"
        "参考上下文：{context}\n\n"
        "请从以下维度评估（1-5分）：\n"
        "1. 忠实度：回答是否完全基于上下文，有无编造信息\n"
        "2. 相关性：回答是否切中问题要点\n"
        "3. 完整性：回答是否涵盖了问题涉及的所有要点\n"
        "4. 专业性：语气是否友好专业\n\n"
        "请给出综合评分和详细理由。"
    )
    parser = JsonOutputParser(pydantic_object=JudgeScore)
    chain = judge_prompt | llm | parser
    result = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context
    })
    return JudgeScore(**result)


test_cases = [
    {"question": "我想退货，需要满足什么条件？", "relevant_keys": ["退货政策"]},
    {"question": "加急配送多久能到？多少钱？", "relevant_keys": ["配送时效"]},
    {"question": "花呗分期有什么条件？", "relevant_keys": ["支付方式"]},
    {"question": "金卡会员有什么权益？", "relevant_keys": ["会员权益"]},
    {"question": "怎么开增值税专用发票？", "relevant_keys": ["发票开具"]},
    {"question": "你们支持货到付款吗？", "relevant_keys": []},
]

print("=" * 70)
print("客服Agent评测报告 - LLM-as-a-Judge")
print("=" * 70)

total_score = 0
for i, tc in enumerate(test_cases):
    result = customer_service_agent(tc["question"])
    judge_result = llm_as_judge(result["question"], result["answer"], result["context"])
    total_score += judge_result.score

    print(f"\n--- 测试用例 {i+1} ---")
    print(f"问题: {tc['question']}")
    print(f"回答: {result['answer'][:100]}...")
    print(f"评分: {judge_result.score}/5")
    print(f"理由: {judge_result.reasoning}")

avg_score = total_score / len(test_cases)
print(f"\n{'='*70}")
print(f"平均得分: {avg_score:.2f}/5")
print(f"评测结论: {'优秀' if avg_score >= 4 else '良好' if avg_score >= 3 else '需改进'}")
print(f"{'='*70}")


class CustomMetric:
    def __init__(self, name: str):
        self.name = name
        self.scores = []

    def measure(self, question: str, answer: str, context: str) -> float:
        raise NotImplementedError

    def average(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0


class FaithfulnessMetric(CustomMetric):
    def __init__(self):
        super().__init__("忠实度")

    def measure(self, question: str, answer: str, context: str) -> float:
        prompt = ChatPromptTemplate.from_template(
            "请评估以下回答的忠实度。忠实度指回答是否完全基于给定的上下文，没有编造信息。\n\n"
            "上下文：{context}\n回答：{answer}\n\n"
            "请给出0到1之间的分数。1表示完全忠实，0表示完全编造。只输出数字。"
        )
        chain = prompt | llm | StrOutputParser()
        score_text = chain.invoke({"context": context, "answer": answer})
        try:
            score = float(score_text.strip())
        except ValueError:
            score = 0.5
        self.scores.append(score)
        return score


class RelevanceMetric(CustomMetric):
    def __init__(self):
        super().__init__("相关性")

    def measure(self, question: str, answer: str, context: str) -> float:
        prompt = ChatPromptTemplate.from_template(
            "请评估以下回答与问题的相关性。相关性指回答是否真正回答了用户的问题。\n\n"
            "问题：{question}\n回答：{answer}\n\n"
            "请给出0到1之间的分数。1表示完全相关，0表示完全不相关。只输出数字。"
        )
        chain = prompt | llm | StrOutputParser()
        score_text = chain.invoke({"question": question, "answer": answer})
        try:
            score = float(score_text.strip())
        except ValueError:
            score = 0.5
        self.scores.append(score)
        return score


class CompletenessMetric(CustomMetric):
    def __init__(self):
        super().__init__("完整性")

    def measure(self, question: str, answer: str, context: str) -> float:
        prompt = ChatPromptTemplate.from_template(
            "请评估以下回答的完整性。完整性指回答是否涵盖了上下文中与问题相关的所有要点。\n\n"
            "问题：{question}\n上下文：{context}\n回答：{answer}\n\n"
            "请给出0到1之间的分数。1表示完全完整，0表示严重遗漏。只输出数字。"
        )
        chain = prompt | llm | StrOutputParser()
        score_text = chain.invoke({"question": question, "context": context, "answer": answer})
        try:
            score = float(score_text.strip())
        except ValueError:
            score = 0.5
        self.scores.append(score)
        return score


print("\n\n" + "=" * 70)
print("客服Agent评测报告 - 多维度自定义指标")
print("=" * 70)

metrics = [FaithfulnessMetric(), RelevanceMetric(), CompletenessMetric()]

for i, tc in enumerate(test_cases):
    result = customer_service_agent(tc["question"])
    print(f"\n--- 测试用例 {i+1}: {tc['question']} ---")
    for metric in metrics:
        score = metric.measure(result["question"], result["answer"], result["context"])
        print(f"  {metric.name}: {score:.2f}")

print(f"\n{'='*70}")
print("各指标平均分:")
for metric in metrics:
    print(f"  {metric.name}: {metric.average():.2f}")
print(f"{'='*70}")
