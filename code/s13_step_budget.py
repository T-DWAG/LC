"""s13_step_budget.py — Agent 的步数预算：查不到时不要无限重试，留出最后一步诚实收尾。

项目根目录运行：python code/s13_step_budget.py
需要模型服务，且服务端支持工具调用（见 s04、s06）。

线上原型：知识库里没有答案，模型却不断换关键词再搜，一次请求烧掉几十次调用。
做法：用 RemainingSteps 在撞上限之前改道到「收尾节点」，告诉用户实情，
而不是等 GraphRecursionError 抛出来变成 500。
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

load_dotenv(override=True)
SEARCH_CALLS = {"n": 0}


@tool
def search_kb(query: str) -> str:
    """在知识库中搜索。"""
    SEARCH_CALLS["n"] += 1
    return "未找到相关条目，请换一个关键词再试。"  # 模拟：库里确实没有


llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)
llm_tools = llm.bind_tools([search_kb])


class State(MessagesState):
    remaining_steps: RemainingSteps  # 框架维护：距离 recursion_limit 还剩几步


def agent(state: State) -> dict:
    return {"messages": [llm_tools.invoke(state["messages"])]}


def route(state: State) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "done"
    # 一轮 = agent + tools 两步；剩余不足以再走一轮就改道，别等异常。
    return "wrap_up" if state["remaining_steps"] <= 3 else "tools"


def wrap_up(state: State) -> dict:
    # 不带工具再问一次：只能基于已有信息作答，并如实说明查不到。
    note = SystemMessage(content="查询次数已用完。请根据已有结果如实告诉用户没有查到，不要编造。")
    return {"messages": [llm.invoke([*state["messages"][:-1], note])]}


graph = (
    StateGraph(State)
    .add_node("agent", agent)
    .add_node("tools", ToolNode([search_kb]))
    .add_node("wrap_up", wrap_up)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", route, {"tools": "tools", "wrap_up": "wrap_up", "done": END})
    .add_edge("tools", "agent")
    .add_edge("wrap_up", END)
    .compile()
)


if __name__ == "__main__":
    ask = [SystemMessage(content="先用 search_kb 查询再回答；查不到就换关键词继续查。"), HumanMessage(content="公司年假怎么算？")]
    out = graph.invoke({"messages": ask}, {"recursion_limit": 10})
    print("搜索调用次数：", SEARCH_CALLS["n"])
    print("最终回答：", out["messages"][-1].content)
