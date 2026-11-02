import os

from dotenv import load_dotenv # 加载环境变量
from langchain_openai import ChatOpenAI # 导入OpenAI模型
from pydantic import SecretStr 


load_dotenv(override=True) # 项目 .env 覆盖系统里已有的同名变量

def make_llm(*,temperature: float = 0.0) -> ChatOpenAI:

    keys=os.getenv("OPENAI_API_KEY")
    base=os.getenv("OPENAI_BASE_URL")
    model=os.getenv("OPENAI_MODEL")


    if not keys:
        raise ValueError("OPENAI_API_KEY is not set")
    if not base:
        raise ValueError("OPENAI_BASE_URL is not set")
    if not model:
        raise ValueError("OPENAI_MODEL is not set")

    return ChatOpenAI(api_key=SecretStr(keys), base_url=base, model=model, temperature=temperature)


if __name__ == "__main__":
    llm = make_llm()
    msg=llm.invoke("Hello, how are you?")
    print(msg.content)

    # 展示llm的配置
    print(llm.model_name)
    print(llm.openai_api_base)
    print(llm.openai_api_key)
    print(llm.temperature)
  
