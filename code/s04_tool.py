import os 

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # OpenAI的模型
from langgraph.prebuilt import create_react_agent  # 基于ReAct策略的Agent
from langchain_core.tools import tool  # 工具


load_dotenv(override=True)  # 项目 .env 覆盖系统里已有的同名变量

def make_llm(*,temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model =os.environ["OPENAI_MODEL"],
        api_key =os.environ["OPENAI_API_KEY"],
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

llm = make_llm()
agent = create_react_agent(llm, tools=[add, multiply])

result =agent.invoke(
    {"messages": [{"role": "user", "content": "先算 3+4，再把结果乘以 2，只给最终数字"}]}
)
for m in result["messages"]:
   m.pretty_print()