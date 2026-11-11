"""s13_skills_deepagents.py — 用官方组件做 Skills：deepagents 的 SkillsMiddleware，读同一组 SKILL.md。

项目根目录运行（两种模式）：
  python code/s13_skills_deepagents.py        完整模式：create_deep_agent(skills=[...])，一个参数开启技能
  python code/s13_skills_deepagents.py lite   精简模式：普通 create_agent + 两个中间件（只给 read_file）
自己逐句提问，直接回车退出。需要 .env（见 s00_env.md）和 pip install deepagents。

和 s13_skills.py 手写版的对照（左边手写，右边 deepagents）：
  - parse_skill 切 frontmatter  →  SkillsMiddleware 按 Agent Skills 规范解析（yaml，校验 name/description）
  - SkillMiddleware 拼技能目录  →  同一个 SkillsMiddleware 把 name + description + 文件路径注入系统提示词
  - 自定义 load_skill 工具      →  通用 read_file 工具，模型按目录里的路径去读 SKILL.md
  - 只有第 2 层（正文）         →  还有第 3 层：SKILL.md 里引用的 scripts/、references/ 也能按需读
文件是通过 backend 读的：FilesystemBackend(root_dir=code/, virtual_mode=True) 把 code/ 当成虚拟根目录 /，
模型看到的路径是 /skills/travel/SKILL.md，越不出 code/。
默认模式用的是完整的 deep agent：自带 todo、子 agent、写文件等工具和很长的系统提示词，token 明显更多；
这里用 permissions 禁止写文件。只想要 Skills 时用 lite 模式，按官方说明自己把中间件装进 create_agent。
"""
import os
import sys
from pathlib import Path

from deepagents import FilesystemMiddleware, FilesystemPermission, create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

load_dotenv(override=True)

CODE_DIR = Path(__file__).resolve().parent
backend = FilesystemBackend(root_dir=CODE_DIR, virtual_mode=True)  # code/ → 虚拟路径 /
SOURCES = ["/skills/"]  # 指向「放技能文件夹的目录」，不是某一个技能

model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)
BASE_PROMPT = "你是公司行政助手。技能里没写的内容，说不知道，请联系 HR，不要编。"


def build_agent(lite: bool):
    if lite:
        # 官方说明：手动组合时要带 FilesystemMiddleware（提供 read_file），SkillsMiddleware 放在它后面
        return create_agent(
            model,
            system_prompt=BASE_PROMPT,
            middleware=[
                FilesystemMiddleware(backend=backend, tools=["read_file"]),
                SkillsMiddleware(backend=backend, sources=SOURCES),
            ],
            checkpointer=InMemorySaver(),
        )
    return create_deep_agent(
        model=model,
        system_prompt=BASE_PROMPT,
        backend=backend,
        skills=SOURCES,
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],  # 只读
        checkpointer=InMemorySaver(),
    )


def tool_calls_this_turn(messages: list, seen: int) -> list[str]:
    """列出本轮模型调用过的工具，方便看它读了哪个 SKILL.md。"""
    calls = []
    for m in messages[seen:]:
        if isinstance(m, AIMessage):
            calls += [f"{c['name']}({c['args'].get('file_path', '')})".replace("()", "") for c in m.tool_calls]
    return calls


if __name__ == "__main__":
    lite = sys.argv[1:] == ["lite"]
    agent = build_agent(lite)
    names = sorted(p.parent.name for p in (CODE_DIR / "skills").glob("*/SKILL.md"))
    print(f"模式：{'create_agent + SkillsMiddleware' if lite else 'create_deep_agent'}。技能：{', '.join(names)}。直接回车退出。")
    print("试试：「发票抬头是什么」「去上海出差能住多贵的酒店」「帮我给 A100 的客户写一段退款成功的回复」「年假几天」")
    config: RunnableConfig = {"configurable": {"thread_id": "skills-deepagents-demo"}}
    while True:
        question = input("\n问题：").strip()
        if not question:
            break
        seen = len(agent.get_state(config).values.get("messages", []))
        usage = UsageMetadataCallbackHandler()
        out = agent.invoke({"messages": [{"role": "user", "content": question}]}, {**config, "callbacks": [usage]})
        print(f"      工具调用：{tool_calls_this_turn(out['messages'], seen) or '无'}")
        print("回答：", out["messages"][-1].content)
        inp = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
        total = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        print(f"本轮 input {inp} tokens，total {total} tokens")

