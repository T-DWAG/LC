"""s09_refund_agent.py — 退款申请演示：提取、查单、规则、审批、模拟退款。

项目根目录运行：python code/s09_refund_agent.py
本例使用模拟订单和退款结果，不连接真实支付系统。
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
    order: NotRequired[Order | None]
    status: NotRequired[str]
    result: NotRequired[str]


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


def extract_request(state: RefundState) -> dict:
    request = extractor.invoke(state["user_text"])
    if not isinstance(request, RefundRequest):
        raise TypeError(f"expected RefundRequest, got {type(request).__name__}")
    return {"order_id": request.order_id, "reason": request.reason}


def route_request(state: RefundState) -> Literal["lookup", "missing"]:
    if not (state.get("order_id") or "").strip() or not (state.get("reason") or "").strip():
        return "missing"
    return "lookup"


def missing_info(state: RefundState) -> dict:
    return {"status": "missing_info", "result": "请提供订单号和退款原因后重新申请。"}


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
builder.add_node("missing", missing_info)
builder.add_node("lookup", check_order)
builder.add_node("policy", check_policy)
builder.add_node("approval", approve_refund)
builder.add_node("refund", refund)
builder.add_node("reject", reject)
builder.add_edge(START, "extract")
builder.add_conditional_edges("extract", route_request)
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

    result = app.invoke({"user_text": text}, config=config)
    if result.get("__interrupt__"):
        print("待审批：", result["__interrupt__"][0].value)
        while True:
            answer = input("批准退款？输入 y 或 n：").strip().lower()
            if answer in ("y", "n"):
                break
            print("请输入 y 或 n。")
        result = app.invoke(Command(resume=(answer == "y")), config=config)
    print("状态：", result["status"])
    print("结果：", result["result"])
