"""s15_token_budget.py — 成本护栏：按会话累计 token，超预算就停止调用模型。

项目根目录运行：python code/s15_token_budget.py
需要模型服务（.env 见 s00_env.md）。

线上原型：一个会话可以无限聊下去，或被脚本刷；没有任何一层知道「这个会话已经花了多少」。
做法：每次调用后把 usage_metadata 累加进状态（随 checkpoint 持久），
调用前先检查预算，超了就走拒绝分支，不再调模型。
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
BUDGET = 400  # 演示用的小额度；真实系统按套餐/用户等级设置

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)


class State(MessagesState):
    tokens_used: Annotated[int, operator.add]  # 累加器：每个节点只返回「本次增量」


def guard(state: State) -> Command[Literal["chat", "refuse"]]:
    return Command(goto="refuse" if state.get("tokens_used", 0) >= BUDGET else "chat")


def chat(state: State) -> dict:
    reply = llm.invoke(state["messages"])
    # 兼容服务端不返回用量的情况：取不到就记 0，并不会让流程中断。
    used = (reply.usage_metadata or {}).get("total_tokens", 0)
    return {"messages": [reply], "tokens_used": used}


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
        out = graph.invoke({"messages": [HumanMessage(content="用三句话介绍一下退款政策的常见要点。")]}, config)
        print(f"第 {i} 问  累计 {out.get('tokens_used', 0)} / {BUDGET} tokens  →  {str(out['messages'][-1].content)[:30]}…")
