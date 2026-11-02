import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)  # 项目 .env 覆盖系统里已有的同名变量


def make_llm(*, temperature: float = 0.0) -> ChatOpenAI:
    if not os.environ["OPENAI_API_KEY"]:
        raise ValueError("OPENAI_API_KEY is not set")
    if not os.environ["OPENAI_BASE_URL"]:
        raise ValueError("OPENAI_BASE_URL is not set")
    if not os.environ["OPENAI_MODEL"]:
        raise ValueError("OPENAI_MODEL is not set")
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        temperature=temperature,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        base_url=os.environ["OPENAI_BASE_URL"],
    )


class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]
    polished: NotRequired[str]


def as_text(content: str | list[str | dict]) -> str:
    if not isinstance(content, str):
        raise TypeError(f"expected str, got {type(content).__name__}")
    return content.strip()


llm = make_llm()


def generate_topic(state: State) -> State:
    return {"topic": state.get("topic") or "socks in the dryer"}


def write_joke(state: State) -> State:
    topic = state.get("topic")
    if not topic:
        raise ValueError("topic is not set")
    msg = llm.invoke(f"Write a one-line joke about {topic}. Reply with the joke only.")
    return {"joke": as_text(msg.content)}


def polish(state: State) -> State:
    joke = state.get("joke")
    if not joke:
        raise ValueError("joke is not set")
    msg = llm.invoke(
        "Rewrite this joke so it is shorter and funnier. Reply with the joke only.\n" + joke
    )
    return {"polished": as_text(msg.content)}


graph = StateGraph(State)
graph.add_node("generate_topic", generate_topic)
graph.add_node("write_joke", write_joke)
graph.add_node("polish", polish)
graph.add_edge(START, "generate_topic")
graph.add_edge("generate_topic", "write_joke")
graph.add_edge("write_joke", "polish")
graph.add_edge("polish", END)

result = graph.compile().invoke({"topic":"电梯里的陌生人"})
print("topic:", result["topic"])
print("joke:", result["joke"])
print("polished:", result["polished"])
