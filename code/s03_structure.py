import os 
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv(override=True)


def make_llm(*,temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        temperature=temperature,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        base_url=os.environ["OPENAI_BASE_URL"],
    )


class Brief(BaseModel):
    title:str =Field(description="短标题")
    points:list[str]=Field(description="3条要点")

llm=make_llm().with_structured_output(Brief)
result=llm.invoke("请总结一下《三体》系列小说的核心内容")
if not isinstance(result, Brief):
    raise TypeError(f"expected Brief, got {type(result).__name__}")
print("title:",result.title)
for i,p in enumerate(result.points,1):
    print(f"{i}.{p}")
