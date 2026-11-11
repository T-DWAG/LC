"""s16_a2a.py — 编排方：从注册表读 Agent Card，把远程 A2A agent 包成工具，交给主 agent 调度。

先在一个终端运行 python code/s16_a2a_agents.py，再在另一个终端运行：
  python code/s16_a2a.py          以 alice 身份提问（有权限）
  python code/s16_a2a.py mallory  以 mallory 身份提问（远程 agent 会拒绝）
自己逐句提问，直接回车退出。

和 s15 supervisor 的区别只在 ask_remote：本地 invoke 换成了 A2A 的 SendMessage。
  - 能力发现：工具的名字和描述来自对方的 Agent Card，编排方不写死；
  - 鉴权：httpx 客户端统一带 Bearer 令牌；启动时故意不带令牌试一次，看到 401；
  - 身份透传：真实用户放进 SendMessageRequest.metadata.user，由对方决定是否放行；
  - 多轮：每个远程 agent 一个 context_id，同一会话内对方记得上文；
  - 超时：每次远程调用 60 秒上限，失败时把错误文字交给主 agent，而不是让整轮崩掉。
"""
import asyncio
import os
import sys
import uuid

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from a2a.client import A2ACardResolver, A2AClientError, AgentCardResolutionError, Client, ClientConfig, ClientFactory
from a2a.helpers import get_message_text, new_text_part
from a2a.types.a2a_pb2 import AgentCard, Message, Role, SendMessageRequest, TaskState

load_dotenv(override=True)

REGISTRY = ["http://127.0.0.1:9101", "http://127.0.0.1:9102"]  # 平台的注册中心；这里写死两个地址
TOKEN = "demo-token"
USER = sys.argv[1] if len(sys.argv) > 1 else "alice"

model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)


async def send(client: Client, text: str, context_id: str) -> str:
    """发一条消息，等 task 结束，按状态取回文字。"""
    request = SendMessageRequest(
        message=Message(message_id=uuid.uuid4().hex, role=Role.ROLE_USER, parts=[new_text_part(text)], context_id=context_id),
        metadata={"user": USER},  # 透传真实发起人
    )
    async for response in client.send_message(request):
        task = response.task
        state = TaskState.Name(task.status.state)
        print(f"      ← task {task.id[:8]} {state}")
        if task.status.state == TaskState.TASK_STATE_COMPLETED:
            return "\n".join(p.text for a in task.artifacts for p in a.parts)
        return f"远程 agent 未完成（{state}）：{get_message_text(task.status.message)}"
    return "远程 agent 没有返回结果"


def make_tool(card: AgentCard, client: Client) -> StructuredTool:
    """用 Agent Card 生成工具：name、description 都来自对方的自我描述。"""
    context_id = uuid.uuid4().hex  # 本次会话与该 agent 的上下文

    async def ask_remote(query: str) -> str:
        print(f"    → A2A {card.name}：{query}")
        try:
            return await asyncio.wait_for(send(client, query, context_id), timeout=60)
        except (A2AClientError, TimeoutError) as e:
            return f"调用 {card.name} 失败：{type(e).__name__} {e}"

    skills = "；".join(f"{s.name}：{s.description}（例：{'、'.join(s.examples)}）" for s in card.skills)
    return StructuredTool.from_function(
        coroutine=ask_remote,
        name=card.name.replace("-", "_"),
        description=f"{card.description} 技能：{skills}。query 写成完整、独立的问题。",
    )


async def main() -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(headers=headers, timeout=60) as http, httpx.AsyncClient() as anonymous:
        tools = []
        for url in REGISTRY:
            try:
                card = await A2ACardResolver(http, url).get_agent_card()  # GET /.well-known/agent-card.json
            except AgentCardResolutionError:
                sys.exit(f"连不上 {url}，请先在另一个终端运行 python code/s16_a2a_agents.py")
            print(f"发现 {card.name} v{card.version}（{card.provider.organization}）：{[s.name for s in card.skills]}")
            tools.append(make_tool(card, ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(card)))

        # 不带令牌试一次，确认对方真的在鉴权
        bare = ClientFactory(ClientConfig(streaming=False, httpx_client=anonymous)).create(card)
        try:
            await send(bare, "hi", uuid.uuid4().hex)
        except A2AClientError as e:
            print(f"不带令牌调用被拒：{type(e).__name__} {str(e)[:60]}")

        agent = create_agent(
            model,
            tools=tools,
            system_prompt="你是公司助手。问题交给对应的远程 agent 回答（可以同时问多个），整理成一段回复。不要自己编数据；远程 agent 拒绝时如实告诉用户。",
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": f"s16-{USER}"}}
        print(f"\n当前用户：{USER}。试试「去上海出差住多少钱？发票抬头写什么？」，再问「那如果去成都呢」。直接回车退出。")
        while True:
            question = input("\n问题：").strip()
            if not question:
                break
            out = await agent.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
            print("回答：", out["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

