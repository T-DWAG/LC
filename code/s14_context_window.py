"""s14_context_window.py — 长对话：只给模型看最近的窗口，关键事实单独「钉」在状态里。

项目根目录运行：python code/s14_context_window.py
需要模型服务（.env 见 s00_env.md）。

线上原型：客服会话越聊越长，全量历史每轮都发给模型，成本和延迟线性上涨，最终超出上下文。
直接截断又会把开头的关键信息（订单号）丢掉。
做法：checkpoint 里保留完整历史（审计、回放要用），但每次调用前只取最近 N 条；
订单号这类不能丢的事实存成独立字段，每轮拼进 system 提示。
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import SecretStr
from typing_extensions import NotRequired

load_dotenv(override=True)
WINDOW = 4  # 只带最近 4 条消息；真实系统常按 token 数裁剪

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)


class State(MessagesState):
    pinned_order: NotRequired[str]  # 不能随窗口滑走的事实
    sent_count: NotRequired[int]  # 记录最近一次实际发给模型的消息条数


def chat(state: State) -> dict:
    recent = state["messages"][-WINDOW:]
    system = "你是客服。简短回答。"
    if state.get("pinned_order"):
        system += f" 已确认的订单号：{state['pinned_order']}。"
    reply = llm.invoke([SystemMessage(content=system), *recent])
    return {"messages": [reply], "sent_count": len(recent)}


graph = StateGraph(State).add_node("chat", chat).add_edge(START, "chat").compile(checkpointer=InMemorySaver())


def say(config: RunnableConfig, text: str, **extra) -> str:
    out = graph.invoke({"messages": [HumanMessage(content=text)], **extra}, config)
    return str(out["messages"][-1].content)


if __name__ == "__main__":
    for pinned in (False, True):
        config: RunnableConfig = {"configurable": {"thread_id": f"pinned-{pinned}"}}
        say(config, "我的订单号是 A100，屏幕碎了。", **({"pinned_order": "A100"} if pinned else {}))
        for filler in ("你们几点下班？", "运费谁承担？", "能开发票吗？", "退款多久到账？"):
            say(config, filler)
        answer = say(config, "我刚才说的订单号是多少？")
        total = len(graph.get_state(config).values["messages"])
        sent = graph.get_state(config).values["sent_count"]
        print(f"钉住订单号={pinned}：历史共 {total} 条，最后一轮只发给模型 {sent} 条\n  回答：{answer}")
