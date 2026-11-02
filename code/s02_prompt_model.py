import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

load_dotenv(override=True) # 项目 .env 覆盖系统里已有的同名变量


def make_llm(*,temperature: float = 0.0) -> ChatOpenAI:
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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "explain the concept of {input}"),
])

llm = make_llm()
chain = prompt | llm 
response = chain.invoke({"input": " what is lcel"})
print(response.content)