"""s10_sqlite_resume.py — 持久化检查点：模型起草审批摘要，进程退出后仍可按 thread_id 恢复审批。

用法（项目根目录，分两次运行，中间进程已退出）：
  python code/s10_sqlite_resume.py start  order-1
  python code/s10_sqlite_resume.py resume order-1 y
需要模型服务（.env 见 s00_env.md）。检查点写入 code/s10_demo.sqlite，演示完可直接删除。
"""
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)
DB_PATH = Path(__file__).with_name("s10_demo.sqlite")


def make_llm(*, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=temperature,
    )


class State(TypedDict):
    request: str
    summary: NotRequired[str]
    result: NotRequired[str]


def draft(state: State) -> dict:
    # 模型调用单独成节点：它完成后检查点已保存，恢复时不会重新调用（省钱，也避免摘要变化）。
    msg = make_llm().invoke(f"用一句话向审批人概括这条退款申请，不要编造信息：{state['request']}")
    return {"summary": str(msg.content)}


def approval(state: State) -> Command[Literal["done", "cancel"]]:
    # 恢复时本节点会从头重跑，所以 interrupt 之前不要放有副作用的操作。
    ok = interrupt({"question": "是否批准？", "summary": state["summary"]})
    return Command(goto="done" if ok else "cancel")


def done(state: State) -> dict:
    return {"result": "已批准"}


def cancel(state: State) -> dict:
    return {"result": "已拒绝"}


builder = StateGraph(State)
builder.add_node("draft", draft)
builder.add_node("approval", approval)
builder.add_node("done", done)
builder.add_node("cancel", cancel)
builder.add_edge(START, "draft")
builder.add_edge("draft", "approval")
builder.add_edge("done", END)
builder.add_edge("cancel", END)


if __name__ == "__main__":
    mode, thread_id = sys.argv[1], sys.argv[2]
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # from_conn_string 是上下文管理器：退出 with 时连接关闭。
    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        app = builder.compile(checkpointer=saver)
        if mode == "start":
            result = app.invoke({"request": "订单 A100 商品损坏，申请退款"}, config)
        else:
            result = app.invoke(Command(resume=sys.argv[3] == "y"), config)
        paused = result.get("__interrupt__")
        print("暂停：" if paused else "结果：", paused[0].value if paused else result["result"])
        print("下一步节点：", app.get_state(config).next)
