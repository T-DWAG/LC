"""s14_context_window.py — 对话窗口会丢信息：已核实的事实要独立存储。

项目根目录运行：python code/s14_context_window.py
需模型服务（.env 见 s00_env.md）；订单库为模拟。
真实业务在身份鉴权后才绑定订单；此例仅演示「可见消息」与「业务事实」分离。
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import SecretStr
from typing_extensions import NotRequired

load_dotenv(override=True)
WINDOW = 4  # 演示按条数；生产按 token 裁剪
ORDERS = {"A100"}  # 模拟查库；真实系统还需验证订单归属
PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]-?\d{3}(?!\d)")

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)


class State(MessagesState):
    verified_order: NotRequired[str]
    sent_count: NotRequired[int]


def chat(state: State) -> dict:
    latest = state["messages"][-1]
    if not isinstance(latest, HumanMessage):
        raise TypeError("只接受用户输入")
    # 业务事实由代码查库后写入，不能信任用户或模型自称「已验证」。
    matches = {m.group().replace("-", "").upper() for m in PATTERN.finditer(str(latest.content))}
    found = matches & ORDERS
    verified = next(iter(found)) if len(found) == 1 else state.get("verified_order")
    system = "你是客服；只能确认已核实的订单号，其他信息不知道就说不知道。"
    if verified:
        system += f" 已核实订单号：{verified}。用户询问订单号时直接回答这个已核实的值。"
    recent = state["messages"][-WINDOW:]
    reply = llm.invoke([SystemMessage(content=system), *recent])
    return {"messages": [reply], "sent_count": len(recent), **({"verified_order": verified} if verified else {})}


graph = StateGraph(State).add_node("chat", chat).add_edge(START, "chat").compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config: RunnableConfig = {"configurable": {"thread_id": "window-demo"}}
    for text in ("订单 A100 屏幕碎了", "你们几点下班？", "运费谁承担？", "能开发票吗？", "退款多久到账？", "我刚才的订单号是多少？"):
        out = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    print("保存的历史条数：", len(out["messages"]))
    print("最后一轮发给模型的历史条数：", out["sent_count"])
    print("独立核实的订单号：", out.get("verified_order"))
    print("回答：", out["messages"][-1].content)
