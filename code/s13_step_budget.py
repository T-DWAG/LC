"""s13_step_budget.py — 查不到资料就停止，不把循环次数当成答案质量。

项目根目录运行：python code/s13_step_budget.py
需要模型服务（.env 见 s00_env.md）；知识库工具为模拟。
业务约束：同一问题最多试两个不同检索词；空结果是业务结果，不是让模型无限重试的故障。
"""
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)
MAX_SEARCHES = 2


class SearchPlan(BaseModel):
    query: str = Field(description="与问题最相关的一个检索词")


llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)
planner = llm.with_structured_output(SearchPlan)


class State(TypedDict):
    question: str
    queries: NotRequired[list[str]]
    hits: NotRequired[list[str]]
    answer: NotRequired[str]


def plan(state: State) -> Command[Literal["search", "finish"]]:
    tried = state.get("queries", [])
    if len(tried) >= MAX_SEARCHES:
        return Command(goto="finish")
    result = planner.invoke(
        f"问题：{state['question']}\n已尝试：{tried}。给一个不同的检索词，不要回答问题。"
    )
    if not isinstance(result, SearchPlan):
        raise TypeError("模型未返回检索词")
    query = result.query.strip()
    if not query or query in tried:
        return Command(goto="finish")  # 换汤不换药，停止浪费检索
    return Command(update={"queries": [*tried, query]}, goto="search")


KB = {"发票": "报销发票抬头为星河科技有限公司。"}  # 模拟授权可见的知识库


def search(state: State) -> Command[Literal["plan", "finish"]]:
    query = state["queries"][-1]
    # 真实系统在这里查库，并核对租户权限；空结果与调用故障必须区分。
    hit = next((doc for term, doc in KB.items() if term in query), None)
    if hit:
        return Command(update={"hits": [hit]}, goto="finish")
    return Command(goto="plan")


def finish(state: State) -> dict:
    if not state.get("hits"):
        # 没证据时不让模型用通用知识编造公司制度；用固定模板回应。
        return {"answer": "知识库未找到这项公司制度，请向 HR 确认。"}
    reply = llm.invoke(f"只根据下面的资料回答问题；资料不足就说不知道。\n资料：{state['hits']}\n问题：{state['question']}")
    return {"answer": str(reply.content)}


graph = (
    StateGraph(State)
    .add_node("plan", plan)
    .add_node("search", search)
    .add_node("finish", finish)
    .add_edge(START, "plan")
    .add_edge("finish", END)
    .compile()
)

if __name__ == "__main__":
    for question in ("公司年假怎么算？", "报销发票抬头是什么？"):
        out = graph.invoke({"question": question}, {"recursion_limit": 20})
        print(question, "实际查询：", out.get("queries", []))
        print("回答：", out["answer"])
