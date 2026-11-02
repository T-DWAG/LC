import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr

load_dotenv(override=True)#加载环境变量

# 根据变量创建LLM模型
def make_llm(*,temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model =os.environ["OPENAI_MODEL"],
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        base_url =os.environ["OPENAI_BASE_URL"],
        temperature =temperature,
    )

llm = make_llm()




def call_model(state: MessagesState):
    msgs= [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
    return {"messages": [llm.invoke(msgs)]}

app =(
    StateGraph(MessagesState)
    .add_node("model",call_model)
    .add_edge(START,"model")
    .add_edge("model",END)
    .compile(checkpointer=InMemorySaver())
)


cfg: RunnableConfig = {"configurable": {"thread_id": "123"}}
cfg2: RunnableConfig = {"configurable": {"thread_id": "456"}}


r1=app.invoke({"messages":[HumanMessage(content="call me T-Dwag,remember my name.")]},config=cfg)
print(r1["messages"][-1].content)


r2=app.invoke({"messages":[HumanMessage(content="what is my name?")]},config=cfg)
# r2=app.invoke({"messages":[HumanMessage(content="what is my name?")]},config=cfg2)

print(r2["messages"][-1].content)