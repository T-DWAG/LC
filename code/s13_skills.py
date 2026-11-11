"""s13_skills.py — Skills：把专项知识写成 SKILL.md，模型需要时才加载（渐进式披露）。

项目根目录运行：python code/s13_skills.py （按需加载）
对照运行：      python code/s13_skills.py full   （把所有技能全文塞进系统提示词）
自己逐句提问，直接回车退出。技能在 code/skills/<名字>/SKILL.md，共三个：invoice、travel、refund-reply。
需要模型服务（.env 见 s00_env.md）和 langchain>=1.0。

s04 的工具是「能做什么」，Skill 是「该怎么做」：一段专项提示词 + 领域知识。技能一多，全塞进系统提示词有两个问题：
  1. 每次调模型都要为用不到的技能付 token；
  2. 无关规则互相干扰，模型容易张冠李戴。
做法（官方 Skills 模式）：
  1. 系统提示词里只放「名字 + 一句话描述」的目录，很便宜；
  2. 给模型一个 load_skill 工具，判断需要时自己读全文；
  3. 目录由中间件在每次调模型前注入，新增技能只需要加一个文件夹。
两种模式各跑一遍，比较每轮的 input tokens：只有三个小技能时，全量模式反而更省，因为按需加载要多调一次模型去 load_skill。
技能越多、越长，按需加载越划算；同时规则互不干扰（问年假时不会读到任何技能）。
"""
import os
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

load_dotenv(override=True)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def parse_skill(path: Path) -> dict[str, str]:
    """读取 SKILL.md：--- 之间是 name / description，后面是正文。"""
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = dict(line.split(":", 1) for line in front.strip().splitlines())
    return {"name": meta["name"].strip(), "description": meta["description"].strip(), "body": body.strip()}


SKILLS = {s["name"]: s for s in (parse_skill(p) for p in sorted(SKILLS_DIR.glob("*/SKILL.md")))}


@tool
def load_skill(name: str) -> str:
    """加载一个技能的完整说明。name 必须是「可用技能」目录里的名字。"""
    print(f"      加载技能：{name}")
    skill = SKILLS.get(name)
    if skill is None:
        return f"没有技能 {name!r}，可用：{', '.join(SKILLS)}"
    return skill["body"]


class SkillMiddleware(AgentMiddleware):
    """每次调模型前，把技能目录追加到系统提示词；同时注册 load_skill 工具。"""

    tools = [load_skill]  # 中间件自带的工具，create_agent 会自动加进去

    def __init__(self) -> None:
        catalog = "\n".join(f"- {s['name']}：{s['description']}" for s in SKILLS.values())
        self.addendum = f"\n\n## 可用技能\n{catalog}\n\n回答相关问题前，先用 load_skill 读取对应技能，按技能里的规则回答。"

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        system = SystemMessage(content=blocks + [{"type": "text", "text": self.addendum}])
        return handler(request.override(system_message=system))


model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)
BASE_PROMPT = "你是公司行政助手。技能里没写的内容，说不知道，请联系 HR，不要编。"


def build_agent(full: bool):
    if full:
        # 对照组：所有技能全文直接写进系统提示词，不需要 load_skill。
        everything = "\n\n".join(f"## {s['name']}\n{s['body']}" for s in SKILLS.values())
        return create_agent(model, tools=[], system_prompt=BASE_PROMPT + "\n\n" + everything, checkpointer=InMemorySaver())
    return create_agent(model, tools=[], system_prompt=BASE_PROMPT, middleware=[SkillMiddleware()], checkpointer=InMemorySaver())


if __name__ == "__main__":
    full = sys.argv[1:] == ["full"]
    agent = build_agent(full)
    print(f"模式：{'全部塞进提示词' if full else '按需加载'}。技能：{', '.join(SKILLS)}。直接回车退出。")
    print("试试：「发票抬头是什么」「去上海出差能住多贵的酒店」「帮我给 A100 的客户写一段退款成功的回复」「年假几天」")
    config: RunnableConfig = {"configurable": {"thread_id": "skills-demo"}}
    while True:
        question = input("\n问题：").strip()
        if not question:
            break
        usage = UsageMetadataCallbackHandler()  # 每轮新建，只统计这一轮
        out = agent.invoke({"messages": [{"role": "user", "content": question}]}, {**config, "callbacks": [usage]})
        print("回答：", out["messages"][-1].content)
        inp = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
        total = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        print(f"本轮 input {inp} tokens，total {total} tokens")
