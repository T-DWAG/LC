"""s15_token_budget.py — 记录实际用量，并阻止下一次超预算会话调用。

项目根目录运行：python code/s15_token_budget.py
需要模型服务（.env 见 s00_env.md）。这里只演示单进程/单线程软限额，
真实的用户/租户限额必须在独立计费服务里原子预留、结算；图状态不是并发配额锁。
"""
import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import SecretStr

load_dotenv(override=True)
BUDGET = 400

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)


class State(MessagesState):
    tokens_used: Annotated[int, operator.add]


def guard(state: State) -> Command[Literal["chat", "refuse"]]:
    return Command(goto="refuse" if state.get("tokens_used", 0) >= BUDGET else "chat")


def chat(state: State) -> dict:
    # 只发当前一问；避免历史膨胀使本节的「额度」示例变成上下文窗口示例。
    reply = llm.invoke(state["messages"][-1:])
    usage = reply.usage_metadata
    if not usage or not isinstance(usage.get("total_tokens"), int):
        raise RuntimeError("服务端未返回 total_tokens：无法计费，不能悄悄按 0 放行")
    return {"messages": [reply], "tokens_used": usage["total_tokens"]}


def refuse(state: State) -> dict:
    return {"messages": [AIMessage(content="本次会话额度已用完，请联系人工客服。")]}


graph = (
    StateGraph(State)
    .add_node("guard", guard)
    .add_node("chat", chat)
    .add_node("refuse", refuse)
    .add_edge(START, "guard")
    .compile(checkpointer=InMemorySaver())
)

if __name__ == "__main__":
    config: RunnableConfig = {"configurable": {"thread_id": "budget-demo"}}
    for i in range(1, 7):
        out = graph.invoke({"messages": [HumanMessage(content="用三句话介绍退款政策需要考虑的要点。")]}, config)
        print(f"第 {i} 问：累计 {out.get('tokens_used', 0)}/{BUDGET} tokens → {str(out['messages'][-1].content)[:30]}")
