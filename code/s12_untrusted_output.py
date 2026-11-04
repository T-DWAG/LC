"""s12_untrusted_output.py — 模型输出是「不可信输入」：先校验，错了把原因回灌给模型重试，仍不行转人工。

项目根目录运行：python code/s12_untrusted_output.py
需要模型服务（.env 见 s00_env.md）。

线上原型：模型从用户原话里抽出订单号，下游拿它去查库、退款。
模型会写出库里不存在的号（大小写、横杠、O 和 0 混淆，甚至凭空编造）。
结构化输出只保证「格式对」，不保证「业务上存在」，所以校验必须在代码里做。
"""
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)
KNOWN_ORDERS = {"A100", "B200"}  # 真实系统里是查库
MAX_ATTEMPTS = 3


class Extracted(BaseModel):
    """只从用户原话里提取，不要猜测。"""

    order_id: str | None = Field(description="订单号，原话里没有则为 null")


llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)
extractor = llm.with_structured_output(Extracted)


class State(TypedDict):
    text: str
    attempts: NotRequired[int]
    feedback: NotRequired[str]
    order_id: NotRequired[str]
    status: NotRequired[str]


def normalize(raw: str) -> str:
    # 确定性的格式问题（大小写、横杠、空格）交给代码，不要浪费一次模型调用。
    return raw.strip().upper().replace("-", "").replace(" ", "")


def extract(state: State) -> Command[Literal["extract"]]:
    attempts = state.get("attempts", 0) + 1
    prompt = state["text"]
    if state.get("feedback"):
        # 把「哪里错了」告诉模型，比单纯重试有效得多。
        prompt += f"\n\n上一次的提取被系统拒绝：{state['feedback']}"
    got = extractor.invoke(prompt)
    raw = got.order_id if isinstance(got, Extracted) else None

    if not raw:
        # 原话里就没有订单号：再问模型十次也不会有，该问用户。
        return Command(update={"attempts": attempts, "status": "need_input"}, goto=END)
    candidate = normalize(raw)
    if candidate in KNOWN_ORDERS:
        return Command(update={"attempts": attempts, "order_id": candidate, "status": "ok"}, goto=END)
    if attempts >= MAX_ATTEMPTS:
        # 不要无限纠错：反复失败说明输入本身有问题，该让人看。
        return Command(update={"attempts": attempts, "status": "manual_review"}, goto=END)
    reason = f"订单号 {raw!r} 在系统中不存在。请只返回原话里逐字出现的订单号，没有就返回 null。"
    return Command(update={"attempts": attempts, "feedback": reason}, goto="extract")


graph = StateGraph(State).add_node("extract", extract).add_edge(START, "extract").compile()


if __name__ == "__main__":
    for text in ("我的订单是 a-100 那个，屏幕碎了", "订单 Z999 一直没收到", "帮我退款"):
        out = graph.invoke({"text": text})
        print(f"{text!r}\n  状态={out['status']} 订单号={out.get('order_id')} 模型调用 {out['attempts']} 次")
