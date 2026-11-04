"""s12_untrusted_output.py — 模型负责抽取，代码负责核对来源与查库。

项目根目录运行：python code/s12_untrusted_output.py
需按 s00_env.md 配模型及结构化输出能力。订单库为模拟数据。
"""
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)
ORDERS = {"A100": "paid", "B200": "refunded"}  # 模拟订单库
ORDER_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]\s*-?\s*\d{3}(?!\d)")


class Extracted(BaseModel):
    order_id: str | None = Field(description="用户原话中的订单号；没说则为 null，不得猜测")


llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
).with_structured_output(Extracted)


class State(TypedDict):
    text: str
    candidate: NotRequired[str | None]
    order_id: NotRequired[str]
    status: NotRequired[str]


def extract(state: State) -> dict:
    result = llm.invoke(state["text"])
    if not isinstance(result, Extracted):
        raise TypeError("模型未返回预期结构；由调用方监控/处理，不伪装成查无此单")
    return {"candidate": result.order_id}


def canonical(value: str) -> str:
    return re.sub(r"[\s-]", "", value).upper()


def verify(state: State) -> dict:
    # 提取值必须能在原文中找到证据；结构化输出只保证形状，不保证真实。
    evidence = {canonical(m.group()) for m in ORDER_PATTERN.finditer(state["text"])}
    candidate = canonical(state["candidate"]) if state.get("candidate") else None
    if not evidence:
        return {"status": "need_input"}  # 原话没给订单号，继续问模型也没用
    if candidate not in evidence:
        return {"status": "needs_verification"}  # 模型看错或猜错，交人工核对原话
    if candidate not in ORDERS:
        return {"status": "needs_verification"}  # 用户说了也未必是有效订单
    return {"order_id": candidate, "status": "ok"}


graph = (
    StateGraph(State)
    .add_node("extract", extract)
    .add_node("verify", verify)
    .add_edge(START, "extract")
    .add_edge("extract", "verify")
    .compile()
)

if __name__ == "__main__":
    for text in ("我的订单是 a-100，屏幕碎了", "订单 Z999 一直没收到", "帮我退款"):
        out = graph.invoke({"text": text})
        print(f"{text!r} → {out['status']} {out.get('order_id', '')}")
