from typing import Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# 审批状态：action_details 是待审的动作，status 是还在等、已通过、还是已拒绝。
class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]


# 审批节点。interrupt 把问题抛给外面并停住当前这轮。
# 同一个 thread 再次 invoke(Command(resume=...)) 时，resume 的值变成 decision。
# Command(goto=...) 按这个值跳到 proceed 或 cancel，所以这里不用再加普通边。
def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    decision = interrupt(
        {
            "question": "Do you approve of this action?",
            "details": state["action_details"],
        }
    )
    return Command(goto="proceed" if decision else "cancel")


# 人点了同意：把状态写成 approved。
def proceed_node(state: ApprovalState):
    return {"status": "approved"}


# 人点了拒绝：把状态写成 rejected。
def cancel_node(state: ApprovalState):
    return {"status": "rejected"}


# interrupt 必须有 checkpointer，挂起的位置才有地方记下来。
# InMemorySaver 只存在当前进程里，按 thread_id 分开每一单。
graph = (
    StateGraph(ApprovalState)
    .add_node("approval", approval_node)
    .add_node("proceed", proceed_node)
    .add_node("cancel", cancel_node)
    .add_edge(START, "approval")
    .add_edge("proceed", END)
    .add_edge("cancel", END)
    .compile(checkpointer=InMemorySaver())
)


# 第一单：转账。第一次 invoke 停在 approval，结果里有 __interrupt__，status 还是 pending。
cfg: RunnableConfig = {"configurable": {"thread_id": "123"}}
paused = graph.invoke(
    {"action_details": "Transfer $500", "status": "pending"},
    config=cfg,
)
print("interrupt", paused.get("__interrupt__"))

# 同一个 thread_id 恢复。resume=True 传回 interrupt()，于是走进 proceed。
approved = graph.invoke(Command(resume=True), cfg)
print("after approve:", approved["status"])








# 第二单用另一个 thread_id，和上面那单的存档互不影响。
cfg2: RunnableConfig = {"configurable": {"thread_id": "456"}}
graph.invoke({"action_details": "Delete DB", "status": "pending"}, cfg2)
# resume=False 传回 interrupt()，于是走进 cancel。
rejected = graph.invoke(Command(resume=False), cfg2)
print("after reject:", rejected["status"])
