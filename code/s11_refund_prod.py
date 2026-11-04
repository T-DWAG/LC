"""s11_refund_prod.py — 退款审批的生产化：把 s09 里「演示可以忽略」的问题逐个补上。

项目根目录运行：python code/s11_refund_prod.py
需要模型服务（.env 见 s00_env.md）和 langgraph>=1.2。支付网关是本文件内的模拟类。

生产里最危险的一步不是调模型，而是「审批通过后调支付网关」：
  1. 网关偶发 502，重试就能成功；
  2. 网关已扣款但响应超时——调用方不知道钱到底退了没有；
  3. 审批等了几小时，订单在别处已被退掉；
  4. 模型服务自己也会挂，不应拖垮整条流程。
"""
import copy
import os
from typing import Literal

import openai
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)


# ---------- 外部世界：订单库与支付网关（模拟） ----------
ORDERS0 = {"A100": {"order_id": "A100", "amount": 199, "status": "paid"}}
ORDERS = copy.deepcopy(ORDERS0)


class GatewayUnavailable(Exception):
    """网关明确没处理：502/503/连接被拒。重试安全。"""


class GatewayTimeout(Exception):
    """读超时：网关可能已经处理了。重试只有在幂等键保护下才安全。"""


class FakeGateway:
    def reset(self, mode: str) -> None:
        self.mode, self.calls, self.ledger = mode, 0, {}

    def refund(self, key: str, order_id: str, amount: int) -> dict:
        self.calls += 1
        if self.mode == "down" or (self.mode == "flaky" and self.calls < 3):
            raise GatewayUnavailable("502 Bad Gateway")
        if key in self.ledger:  # 幂等：同一个 key 只生效一次，重复请求返回原结果
            return self.ledger[key]
        self.ledger[key] = {"order_id": order_id, "amount": amount}
        if self.mode == "timeout_after_commit" and self.calls == 1:
            raise GatewayTimeout("read timeout")  # 钱已退，响应丢了
        return self.ledger[key]


GATEWAY = FakeGateway()


# ---------- 模型 ----------
class RefundRequest(BaseModel):
    """提取用户提供的内容，不要猜测未提供的信息。"""

    order_id: str | None = Field(description="订单号；未提供时为 null")
    reason: str | None = Field(description="退款原因；未提供时为 null")


def make_llm(base_url: str | None = None) -> ChatOpenAI:
    # max_retries=0：重试只在图这一层做，避免客户端和图两层叠加成 3×3 次。
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        base_url=base_url or os.environ["OPENAI_BASE_URL"],
        temperature=0,
        max_retries=0,
        timeout=15,
    )


# ---------- 图 ----------
class State(TypedDict):
    user_text: str
    order_id: NotRequired[str]
    reason: NotRequired[str]
    amount: NotRequired[int]
    refund_key: NotRequired[str]
    status: NotRequired[str]
    note: NotRequired[str]


MODEL_URL: str | None = None  # 场景 5 把它指向不可达地址，模拟模型服务宕机


def extract(state: State) -> dict:
    req = make_llm(MODEL_URL).with_structured_output(RefundRequest).invoke(state["user_text"])
    if not isinstance(req, RefundRequest) or not req.order_id or not req.reason:
        raise ValueError("缺少订单号或原因")  # 本地逻辑错误：不重试，直接暴露
    return {"order_id": req.order_id.strip().upper(), "reason": req.reason}


def lookup(state: State) -> Command[Literal["approval", "finish"]]:
    order = ORDERS.get(state["order_id"])
    if order is None or order["status"] != "paid":
        return Command(update={"status": "not_eligible", "note": "订单不存在或不是已支付"}, goto="finish")
    return Command(update={"amount": order["amount"]}, goto="approval")


def approval(state: State) -> Command[Literal["recheck", "finish"]]:
    # 人可能几小时后才回复。快照只是「审批时看到的样子」，不是执行时的事实。
    ok = interrupt({"question": "是否批准退款？", "order_id": state["order_id"], "amount": state["amount"], "reason": state["reason"]})
    if ok is not True:
        return Command(update={"status": "rejected", "note": "审批人拒绝"}, goto="finish")
    return Command(goto="recheck")


def recheck(state: State) -> Command[Literal["charge", "finish"]]:
    # 审批期间订单可能已在别处退掉：执行前必须重新读取最新状态。
    order = ORDERS[state["order_id"]]
    if order["status"] != "paid":
        return Command(update={"status": "stale", "note": "审批期间订单状态已变化，未执行"}, goto="finish")
    # 幂等键必须由业务数据推出。节点恢复时会重跑，若在这里用 uuid4()，每次都是新键，幂等就失效了。
    return Command(update={"refund_key": f"refund-{state['order_id']}"}, goto="charge")


def charge(state: State) -> dict:
    result = GATEWAY.refund(state["refund_key"], state["order_id"], state["amount"])
    ORDERS[state["order_id"]]["status"] = "refunded"
    return {"status": "refunded", "note": f"已退 {result['amount']} 元"}


def charge_unknown(state: State, error: NodeError) -> Command:
    # 重试用尽。注意：此时「钱没退」和「钱退了但没拿到响应」两种可能都存在，
    # 所以不能标记失败，更不能让用户重新申请。只能挂起等对账。
    return Command(
        update={"status": "refund_unknown", "note": f"网关异常（{type(error.error).__name__}），键 {state['refund_key']}，转人工对账"},
        goto="finish",
    )


def extract_failed(state: State, error: NodeError) -> Command:
    # 模型挂了不该让整个退款流程报 500：转人工，把原文留给客服。
    return Command(update={"status": "manual_review", "note": f"模型不可用（{type(error.error).__name__}），原文已留存"}, goto="finish")


def finish(state: State) -> dict:
    return {}


MODEL_ERRORS = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
GATEWAY_ERRORS = (GatewayUnavailable, GatewayTimeout)

builder = (
    StateGraph(State)
    .add_node("extract", extract, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=MODEL_ERRORS), error_handler=extract_failed)
    .add_node("lookup", lookup)
    .add_node("approval", approval)
    .add_node("recheck", recheck)
    .add_node("charge", charge, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=GATEWAY_ERRORS), error_handler=charge_unknown)
    .add_node("finish", finish)
    .add_edge(START, "extract")
    .add_edge("extract", "lookup")
    .add_edge("charge", "finish")
    .add_edge("finish", END)
)
app = builder.compile(checkpointer=InMemorySaver())


def run(title: str, *, mode: str = "ok", stale: bool = False, model_down: bool = False) -> None:
    global MODEL_URL
    GATEWAY.reset(mode)
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(ORDERS0))
    MODEL_URL = "http://127.0.0.1:9/v1" if model_down else None
    config: RunnableConfig = {"configurable": {"thread_id": title}}
    out = app.invoke({"user_text": "订单 A100 商品损坏，申请退款"}, config)
    if out.get("__interrupt__"):
        if stale:
            ORDERS["A100"]["status"] = "refunded"  # 审批等待期间，别处已经退了
        out = app.invoke(Command(resume=True), config)
    print(f"{title}\n  状态={out['status']}  {out.get('note', '')}\n  网关调用 {GATEWAY.calls} 次，实际到账 {len(GATEWAY.ledger)} 笔")


if __name__ == "__main__":
    run("1 网关偶发 502（重试即可）", mode="flaky")
    run("2 已扣款但响应超时（幂等键兜底）", mode="timeout_after_commit")
    run("3 网关持续不可用（转对账）", mode="down")
    run("4 审批期间订单已被退（执行前复核）", stale=True)
    run("5 模型服务宕机（转人工）", model_down=True)
