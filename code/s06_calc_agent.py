import os
import operator  # 导入运算符
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

load_dotenv(override=True)


def make_llm(*,temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model =os.environ["OPENAI_MODEL"],
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        base_url =os.environ["OPENAI_BASE_URL"],
        temperature =temperature,
    )



@tool
def add(a:int,b:int) -> int:
    """add a and b"""
    return a + b

@tool
def multiply(a:int,b:int) -> int:
    """multiply a and b"""
    return a * b

@tool
def divide(a:int,b:int) -> float:
    """divide a by b"""
    return a / b

tools = [add, multiply, divide]

model_with_tools = make_llm().bind_tools(tools)

class MessageState(TypedDict):
    messages:Annotated[list[AnyMessage], operator.add]
    llm_calls:int


def llm_call(state:MessageState) :
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful arithmetic assistant.")] + state["messages"]
            )
        ],
        "llm_calls": state["llm_calls"] + 1,
    }


agent_builder = StateGraph(MessageState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tools", ToolNode(tools))
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", tools_condition)
agent_builder.add_edge("tools", "llm_call")
agent = agent_builder.compile()

init: MessageState = {
    "messages": [HumanMessage(content="Add 3 and 4. Then multiply the result by 2.")],
    "llm_calls": 0,
}
out = agent.invoke(init)
for m in out["messages"]:
    m.pretty_print()
print("llm_calls:", out["llm_calls"])