"""s11_refund_prod.py — 退款审批的生产化：把 s09 里「演示可以忽略」的问题逐个补上。

项目根目录运行：python code/s11_refund_prod.py
选一个场景，输入退款申请（可以留空，缺什么会追问，最多 3 轮），审批时输入 y。终端会逐节点打印执行轨迹，
以及每一次网关 / 模型调用，最后对照该场景的预期。
订单库只有已支付的 A100。需要模型服务（.env 见 s00_env.md）和 langgraph>=1.2。支付网关是本文件内的模拟类。

生产里最危险的一步不是调模型，而是「审批通过后调支付网关」：
  1. 网关明确拒绝请求、未提交，稍后恢复，可安全重试；真实 502 不保证未处理；
  2. 网关已扣款但响应超时——调用方不知道钱到底退了没有；
  3. 网关持续不可用，重试用尽——结果未知，只能转对账；
  4. 审批等了几小时，订单在别处已被退掉；
  5. 模型服务自己也会挂，不应拖垮整条流程。
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
    """模拟网关明确拒绝、确认未提交的请求；真实 502 未必能证明未处理。"""


class GatewayTimeout(Exception):
    """读超时：网关可能已经处理了。重试只有在幂等键保护下才安全。"""


class FakeGateway:
    def reset(self, mode: str) -> None:
        self.mode, self.calls, self.ledger = mode, 0, {}

    def refund(self, key: str, order_id: str, amount: int) -> dict:
        self.calls += 1
        tag = f"      网关 #{self.calls}  key={key}  →"
        if self.mode == "down" or (self.mode == "flaky" and self.calls < 3):
            print(tag, "拒绝，未提交（GatewayUnavailable）")
            raise GatewayUnavailable("请求被明确拒绝，未提交")
        if key in self.ledger:  # 幂等：同一个 key 只生效一次，重复请求返回原结果
            print(tag, "键已存在，返回原结果，不重复退款")
            return self.ledger[key]
        self.ledger[key] = {"order_id": order_id, "amount": amount}
        if self.mode == "timeout_after_commit" and self.calls == 1:
            print(tag, "已入账，但响应超时（GatewayTimeout）")
            raise GatewayTimeout("read timeout")  # 钱已退，响应丢了
        print(tag, f"入账 {amount} 元")
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
    clarify_turns: NotRequired[int]


CLARIFY_TURNS = 3  # 最多追问三轮，仍不全就结束，避免无限来回


MODEL_URL: str | None = None  # 场景 5 把它指向不可达地址，模拟模型服务宕机


def extract(state: State) -> Command[Literal["lookup", "clarify", "finish"]]:
    text = state["user_text"].strip()
    if text:
        print("      模型调用：抽取订单号和原因")
        req = make_llm(MODEL_URL).with_structured_output(RefundRequest).invoke(text)
    else:
        req = RefundRequest(order_id=None, reason=None)  # 什么都没说，不必调模型，直接追问
    if not isinstance(req, RefundRequest):
        raise TypeError(f"结构化输出类型异常：{type(req).__name__}")  # 代码 bug：不重试，直接暴露
    # 新一轮抽不到时保留上一轮已拿到的字段。
    order_id = (req.order_id or "").strip().upper() or state.get("order_id")
    reason = req.reason or state.get("reason")
    update = {k: v for k, v in (("order_id", order_id), ("reason", reason)) if v}
    if order_id and reason:
        return Command(update=update, goto="lookup")
    # 用户没给全信息是正常业务分支，不是异常：先追问，不进重试也不进 error_handler。
    if state.get("clarify_turns", 0) < CLARIFY_TURNS:
        return Command(update=update, goto="clarify")
    missing = "和".join(name for name, v in (("订单号", order_id), ("退款原因", reason)) if not v)
    return Command(update={**update, "status": "need_input", "note": f"追问 {CLARIFY_TURNS} 轮仍缺{missing}"}, goto="finish")


def clarify(state: State) -> Command[Literal["extract"]]:
    # interrupt 单独放一个节点：恢复时只重跑这里，不会重新调用模型。
    missing = [name for name, key in (("订单号", "order_id"), ("退款原因", "reason")) if not state.get(key)]
    reply = interrupt({"kind": "clarify", "missing": missing, "turn": state.get("clarify_turns", 0) + 1})
    # 把补充内容拼到原文后面，让模型基于完整上下文重新抽取。
    text = "\n".join(part for part in (state["user_text"].strip(), str(reply).strip()) if part)
    return Command(update={"user_text": text, "clarify_turns": state.get("clarify_turns", 0) + 1}, goto="extract")


def lookup(state: State) -> Command[Literal["approval", "finish"]]:
    order = ORDERS.get(state["order_id"])
    if order is None or order["status"] != "paid":
        return Command(update={"status": "not_eligible", "note": "订单不存在或不是已支付"}, goto="finish")
    return Command(update={"amount": order["amount"]}, goto="approval")


def approval(state: State) -> Command[Literal["recheck", "finish"]]:
    # 人可能几小时后才回复。快照只是「审批时看到的样子」，不是执行时的事实。
    ok = interrupt({"kind": "approval", "order_id": state["order_id"], "amount": state["amount"], "reason": state["reason"]})
    if ok is not True:
        return Command(update={"status": "rejected", "note": "审批人拒绝"}, goto="finish")
    return Command(goto="recheck")


def recheck(state: State) -> Command[Literal["charge", "finish"]]:
    # 审批期间订单可能已在别处退掉：先读最新状态；真实系统还要在执行时做原子校验。
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
    if not isinstance(error.error, GATEWAY_ERRORS):
        raise error.error  # 本地 bug 不冒充「网关异常」
    # 重试用尽。注意：此时「钱没退」和「钱退了但没拿到响应」两种可能都存在，
    # 所以不能标记失败，更不能让用户重新申请。只能挂起等对账。
    return Command(
        update={"status": "refund_unknown", "note": f"网关异常（{type(error.error).__name__}），键 {state['refund_key']}，转人工对账"},
        goto="finish",
    )


def extract_failed(state: State, error: NodeError) -> Command:
    # error_handler 会收到节点的所有异常，不只是 retry_on 里的。
    # 本地 bug（如上面的 TypeError）原样抛出，只有模型服务故障才转人工。
    if not isinstance(error.error, MODEL_ERRORS):
        raise error.error
    # 模型挂了不该让整个退款流程报 500：转人工，把原文留给客服。
    return Command(update={"status": "manual_review", "note": f"模型不可用（{type(error.error).__name__}），原文已留存"}, goto="finish")


def finish(state: State) -> dict:
    return {}


MODEL_ERRORS = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
GATEWAY_ERRORS = (GatewayUnavailable, GatewayTimeout)

builder = (
    StateGraph(State)
    .add_node("extract", extract, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=MODEL_ERRORS), error_handler=extract_failed)
    .add_node("clarify", clarify)
    .add_node("lookup", lookup)
    .add_node("approval", approval)
    .add_node("recheck", recheck)
    .add_node("charge", charge, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=GATEWAY_ERRORS), error_handler=charge_unknown)
    .add_node("finish", finish)
    .add_edge(START, "extract")
    .add_edge("charge", "finish")
    .add_edge("finish", END)
)
app = builder.compile(checkpointer=InMemorySaver())


def trace(stream) -> dict | None:
    """逐个打印节点的执行结果；返回中断 payload（没有暂停则为 None）。"""
    paused = None
    for chunk in stream:  # 把流消费完，让本次 invoke 正常收尾、写完检查点
        for node, update in chunk.items():
            if node == "__interrupt__":
                paused = update[0].value
                continue
            name = node.replace("__error_handler__", "") + (" · error_handler" if node.startswith("__error_handler__") else "")
            print(f"  ✓ {name:<24} {update or ''}", flush=True)
    return paused


def run(title: str, expect: str, *, mode: str = "ok", stale: bool = False, model_down: bool = False) -> None:
    global MODEL_URL
    GATEWAY.reset(mode)
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(ORDERS0))
    MODEL_URL = "http://127.0.0.1:9/v1" if model_down else None
    text = input("退款申请（例：订单 A100 的耳机有杂音）：").strip()
    config: RunnableConfig = {"configurable": {"thread_id": title}}

    print("\n[invoke #1] 从 START 开始跑")
    payload = trace(app.stream({"user_text": text}, config, stream_mode="updates"))
    step = 1
    while payload is not None:  # 每次 interrupt 都是一次「暂停 → 等人 → resume」
        step += 1
        if payload["kind"] == "clarify":
            print(f"  ⏸ clarify 暂停（interrupt），第 {payload['turn']}/{CLARIFY_TURNS} 次追问，缺少：{'、'.join(payload['missing'])}")
            resume = input(f"\n>>> 你现在是用户，请补充{'和'.join(payload['missing'])}：").strip()
        else:
            print("  ⏸ approval 暂停（interrupt），图状态已存进检查点，等审批人回复：")
            print(f"      订单 {payload['order_id']}  金额 {payload['amount']} 元  原因：{payload['reason']}")
            if stale:  # 模拟「审批等了几小时」：这期间订单在别处被退掉
                ORDERS[payload["order_id"]]["status"] = "refunded"
                print(f"  ⚡ 审批等待期间，订单 {payload['order_id']} 已在别处退款（检查点里的快照还是 paid）")
            while (answer := input("\n>>> 你现在是审批人，批准退款吗？输入 y 或 n 后回车：").strip().lower()) not in ("y", "n"):
                print("请输入 y 或 n。")
            resume = answer == "y"
        print(f"\n[invoke #{step}] Command(resume=...) 从 {payload['kind']} 继续")
        payload = trace(app.stream(Command(resume=resume), config, stream_mode="updates"))

    final = app.get_state(config).values
    print(f"\n结果  status={final.get('status')}  {final.get('note', '')}")
    print(f"账本  网关调用 {GATEWAY.calls} 次，实际到账 {len(GATEWAY.ledger)} 笔")
    if final.get("status") in ("need_input", "not_eligible", "rejected"):
        print("提示  这次在本场景要演示的环节之前就结束了，不用对照预期。要看本场景，请给出订单号 A100 和原因，审批输入 y。")
    else:
        print(f"预期  {expect}")

SCENES = {
    "1": ("网关明确未提交（RetryPolicy 重试）", "批准后网关 3 次：前 2 次拒绝，第 3 次入账；refunded，到账 1 笔", {"mode": "flaky"}),
    "2": ("已扣款但响应超时（幂等键兜底）", "批准后网关 2 次：第 1 次已入账但超时，第 2 次同键去重；refunded，到账 1 笔", {"mode": "timeout_after_commit"}),
    "3": ("网关持续不可用（error_handler 转对账）", "批准后网关 3 次都拒绝；refund_unknown，带幂等键转对账，不能标失败", {"mode": "down"}),
    "4": ("审批期间订单已被退（recheck 复核）", "批准后 recheck 发现已不是 paid；stale，网关 0 次", {"stale": True}),
    "5": ("模型服务宕机（error_handler 转人工）", "extract 模型调用 3 次都失败；manual_review，不进审批，网关 0 次", {"model_down": True}),
}


if __name__ == "__main__":
    print("订单库只有已支付的 A100。选一个场景，观察它走了哪条路：")
    for key, (title, _, _) in SCENES.items():
        print(f"  {key}. {title}")
    choice = input("场景编号：").strip()
    if choice not in SCENES:
        raise ValueError("请输入 1 到 5")
    title, expect, options = SCENES[choice]
    run(f"{choice} {title}", expect, **options)
