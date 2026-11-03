"""s09_refund_agent.py — 退款申请演示：提取、追问、查单、规则、审批、模拟退款。

项目根目录运行：python code/s09_refund_agent.py
订单号或原因缺失时追问 3 步，补齐后才查单。本例使用模拟订单和退款结果，不连接真实支付系统。
"""
import os
from datetime import date, timedelta
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)


class RefundRequest(BaseModel):
    """提取用户提供的内容，不要猜测未提供的信息。"""

    order_id: str | None = Field(description="订单号；未提供时为 null")
    reason: str | None = Field(description="退款原因；未提供时为 null")


class Order(TypedDict):
    order_id: str
    amount: int
    status: Literal["paid", "refunded", "unpaid"]
    paid_on: str
    simulate_failure: bool


class RefundState(TypedDict):
    user_text: str
    order_id: NotRequired[str | None]
    reason: NotRequired[str | None]
    clarify_turns: NotRequired[int]
    order: NotRequired[Order | None]
    status: NotRequired[str]
    result: NotRequired[str]


CLARIFY_TURNS = 3


@tool
def lookup_order(order_id: str) -> Order | None:
    """根据订单号查询模拟订单，不存在返回 None。"""
    # 日期相对今天构造，让练习不因日历变化而过期。
    today = date.today()
    orders: dict[str, Order] = {
        "A100": {"order_id": "A100", "amount": 199, "status": "paid", "paid_on": today.isoformat(), "simulate_failure": False},
        "B200": {"order_id": "B200", "amount": 89, "status": "refunded", "paid_on": today.isoformat(), "simulate_failure": False},
        "C300": {"order_id": "C300", "amount": 50, "status": "paid", "paid_on": (today - timedelta(days=31)).isoformat(), "simulate_failure": False},
        "D400": {"order_id": "D400", "amount": 99, "status": "paid", "paid_on": today.isoformat(), "simulate_failure": True},
    }
    return orders.get(order_id.strip().upper())


llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)
extractor = llm.with_structured_output(RefundRequest)


def _filled(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def _complete(state: RefundState) -> bool:
    return _filled(state.get("order_id")) is not None and _filled(state.get("reason")) is not None


def _read_request(text: str) -> RefundRequest:
    request = extractor.invoke(text)
    if not isinstance(request, RefundRequest):
        raise TypeError(f"expected RefundRequest, got {type(request).__name__}")
    return request


def extract_request(state: RefundState) -> dict:
    request = _read_request(state["user_text"])
    return {"order_id": _filled(request.order_id), "reason": _filled(request.reason)}


def route_request(state: RefundState) -> Literal["lookup", "clarify"]:
    if _complete(state):
        return "lookup"
    return "clarify"


def _clarify_question(state: RefundState) -> str:
    turn = state.get("clarify_turns", 0) + 1
    order_id = _filled(state.get("order_id"))
    reason = _filled(state.get("reason"))
    if turn == 1:
        missing = [name for name, value in (("订单号", order_id), ("退款原因", reason)) if value is None]
        if order_id is None:
            return f"信息不完整，缺少{'、'.join(missing)}。第 {turn}/{CLARIFY_TURNS} 步：请回复订单号。"
        return f"信息不完整，缺少退款原因。第 {turn}/{CLARIFY_TURNS} 步：请回复退款原因。"
    if turn == 2:
        if reason is None:
            return f"第 {turn}/{CLARIFY_TURNS} 步：请回复退款原因。当前订单号：{order_id or '仍缺'}。"
        if order_id is None:
            return f"第 {turn}/{CLARIFY_TURNS} 步：请回复订单号。当前原因：{reason}。"
        return f"第 {turn}/{CLARIFY_TURNS} 步：已记录订单 {order_id}，原因「{reason}」。要修改请直接说，否则回复确认。"
    return (
        f"第 {turn}/{CLARIFY_TURNS} 步：请确认订单号 {order_id or '（仍缺）'}，"
        f"退款原因 {reason or '（仍缺）'}。回复确认，或直接改正。"
    )


def clarify(state: RefundState) -> dict:
    reply = interrupt({
        "kind": "clarify",
        "turn": state.get("clarify_turns", 0) + 1,
        "question": _clarify_question(state),
    })
    text = reply.strip() if isinstance(reply, str) else ""
    order_id = _filled(state.get("order_id"))
    reason = _filled(state.get("reason"))
    if text:
        request = _read_request(
            "只提取用户明确说出的订单号和退款原因，没说到的保持为空，不要猜测。\n"
            f"已有订单号：{order_id or '无'}\n"
            f"已有原因：{reason or '无'}\n"
            f"用户补充：{text}"
        )
        order_id = _filled(request.order_id) or order_id
        reason = _filled(request.reason) or reason
    return {
        "user_text": text,
        "order_id": order_id,
        "reason": reason,
        "clarify_turns": state.get("clarify_turns", 0) + 1,
    }


def route_clarify(state: RefundState) -> Literal["clarify", "lookup", "missing"]:
    if state.get("clarify_turns", 0) < CLARIFY_TURNS:
        return "clarify"
    if _complete(state):
        return "lookup"
    return "missing"


def missing_info(state: RefundState) -> dict:
    missing = [name for name, value in (("订单号", _filled(state.get("order_id"))), ("退款原因", _filled(state.get("reason")))) if value is None]
    detail = "、".join(missing) if missing else "必要信息"
    return {"status": "missing_info", "result": f"追问 {CLARIFY_TURNS} 次后仍缺少{detail}，申请结束。"}


def check_order(state: RefundState) -> dict:
    order_id = state.get("order_id")
    if not order_id:
        raise ValueError("order_id is required")
    return {"order": lookup_order.invoke({"order_id": order_id})}


def check_policy(state: RefundState) -> dict:
    """示例规则：订单已支付，且支付日期距今天不超过 30 天。"""
    order = state.get("order")
    if order is None:
        return {"status": "order_not_found", "result": "订单不存在。"}
    if order["status"] != "paid":
        return {"status": "not_eligible", "result": "订单未支付或已退款。"}
    paid_on = date.fromisoformat(order["paid_on"])
    age = (date.today() - paid_on).days
    if age < 0 or age > 30:
        return {"status": "not_eligible", "result": "订单支付日期异常或已超过 30 天退款期限。"}
    return {"status": "eligible", "result": "符合示例退款规则，等待人工审批。"}


def route_policy(state: RefundState) -> Literal["approval", "finish"]:
    return "approval" if state.get("status") == "eligible" else "finish"


def approve_refund(state: RefundState) -> Command[Literal["refund", "reject"]]:
    decision = interrupt({
        "kind": "approval",
        "question": "是否批准退款？",
        "order": state.get("order"),
        "reason": state.get("reason"),
    })
    if not isinstance(decision, bool):
        raise ValueError("审批结果必须是 True 或 False")
    return Command(goto="refund" if decision else "reject")


def refund(state: RefundState) -> dict:
    # 仅模拟执行结果，批准并不等于退款成功。
    order = state.get("order")
    if order is None:
        raise ValueError("order is required")
    if order["simulate_failure"]:
        return {"status": "refund_failed", "result": "审批通过，但模拟退款失败。"}
    return {"status": "refund_succeeded", "result": f"模拟退款成功：{order['order_id']}"}


def reject(state: RefundState) -> dict:
    return {"status": "rejected", "result": "退款申请未获批准。"}


builder = StateGraph(RefundState)
builder.add_node("extract", extract_request)
builder.add_node("clarify", clarify)
builder.add_node("missing", missing_info)
builder.add_node("lookup", check_order)
builder.add_node("policy", check_policy)
builder.add_node("approval", approve_refund)
builder.add_node("refund", refund)
builder.add_node("reject", reject)
builder.add_edge(START, "extract")
builder.add_conditional_edges("extract", route_request)
builder.add_conditional_edges("clarify", route_clarify)
builder.add_edge("lookup", "policy")
builder.add_conditional_edges("policy", route_policy, {"approval": "approval", "finish": END})
for node in ("missing", "refund", "reject"):
    builder.add_edge(node, END)
app = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    text = input("退款申请：").strip()
    if not text:
        raise ValueError("请输入退款申请")

    pending: RefundState | Command = {"user_text": text}
    while True:
        result = app.invoke(pending, config=config)
        paused = result.get("__interrupt__")
        if not paused:
            break
        payload = paused[0].value
        if not isinstance(payload, dict):
            raise TypeError(f"expected interrupt payload dict, got {type(payload).__name__}")
        if payload.get("kind") == "clarify":
            print(payload["question"])
            pending = Command(resume=input("补充：").strip())
            continue
        print("待审批：", payload)
        while True:
            answer = input("批准退款？输入 y 或 n：").strip().lower()
            if answer in ("y", "n"):
                break
            print("请输入 y 或 n。")
        pending = Command(resume=(answer == "y"))
    print("状态：", result["status"])
    print("结果：", result["result"])