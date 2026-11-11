"""s16_a2a_agents.py — 两个「不同团队维护」的 A2A agent 服务：发票 agent（9101）、差旅 agent（9102）。

项目根目录先运行本文件（保持运行），再在另一个终端运行 s16_a2a.py：
  python code/s16_a2a_agents.py
需要 .env（见 s00_env.md）和 pip install "a2a-sdk[http-server]>=1.3" uvicorn。

每个服务对外只暴露两样东西，内部用什么框架、模型、工具，调用方都看不到：
  1. GET /.well-known/agent-card.json  Agent Card：名字、描述、技能、版本、维护人、接口地址；
  2. POST /      JSON-RPC 的 SendMessage：收消息，返回 task（状态 + artifacts）。
本文件演示平台层的三件事：
  - 鉴权：没带 Authorization: Bearer demo-token 的请求直接 401；
  - 身份透传：真实发起人放在请求 metadata.user 里，不在白名单就把 task 置为 REJECTED；
  - 多轮记忆：A2A 的 context_id 当作 LangGraph 的 thread_id，同一个 context 内记得上文。
"""
import asyncio
import os

import uvicorn
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill

load_dotenv(override=True)

TOKEN = "demo-token"  # 演示用；生产环境由网关签发、校验
ALLOWED_USERS = {"alice", "bob"}  # 演示用的「有权限的员工」

model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)


class LangChainExecutor(AgentExecutor):
    """把任意 LangChain agent 接到 A2A：收消息 → 建 task → 调 agent → 写 artifact → 完成。"""

    def __init__(self, name: str, agent) -> None:
        self.name = name
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise ValueError("请求里没有 message")
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)  # 先登记 task，状态 SUBMITTED
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        user = context.metadata.get("user", "")
        question = context.get_user_input()
        print(f"[{self.name}] user={user!r} context={task.context_id[:8]} 问：{question}")
        if user not in ALLOWED_USERS:
            await updater.reject(updater.new_agent_message([new_text_part(f"用户 {user or '(未提供)'} 无权使用 {self.name}")]))
            return
        await updater.start_work()  # WORKING
        config: RunnableConfig = {"configurable": {"thread_id": task.context_id}}
        out = await self.agent.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
        await updater.add_artifact([new_text_part(out["messages"][-1].content)], name="answer")
        await updater.complete()  # COMPLETED；异常会被框架转成 FAILED

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("演示服务不支持取消")


class BearerAuth(BaseHTTPMiddleware):
    """Agent Card 公开可读；JSON-RPC 调用必须带令牌。"""

    async def dispatch(self, request, call_next):
        if request.method == "POST" and request.headers.get("authorization") != f"Bearer {TOKEN}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_app(port: int, name: str, description: str, owner: str, skill: AgentSkill, prompt: str) -> Starlette:
    card = AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        provider=AgentProvider(organization=owner, url="https://example.com"),
        supported_interfaces=[AgentInterface(url=f"http://127.0.0.1:{port}/", protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[skill],
    )
    agent = create_agent(model, tools=[], system_prompt=prompt, checkpointer=InMemorySaver())
    handler = DefaultRequestHandler(agent_executor=LangChainExecutor(name, agent), task_store=InMemoryTaskStore(), agent_card=card)
    app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/"))
    app.add_middleware(BearerAuth)
    return app


APPS = {
    9101: build_app(
        9101, "invoice-agent", "财务部的发票助手：发票抬头、税号、发票类型。", "财务部 · 张三",
        AgentSkill(id="invoice", name="发票咨询", description="回答公司发票抬头、税号和发票类型", tags=["发票", "报销"],
                           examples=["发票抬头写什么？"]),
        "你是发票专员。抬头：星河科技有限公司；税号：91310115MA1K4QXY2B；差旅开普通发票。只回答发票问题，简短。",
    ),
    9102: build_app(
        9102, "travel-agent", "行政部的差旅助手：住宿标准、交通标准。", "行政部 · 李四",
        AgentSkill(id="travel", name="差旅标准", description="回答出差住宿、交通标准", tags=["差旅", "出差"],
                           examples=["去上海出差能住多贵的酒店？"]),
        "你是差旅专员。住宿：一线城市每晚 600 元，其他 400 元；800 公里以上可坐飞机经济舱。"
        "记住用户在本次会话里说过的城市。只回答差旅问题，简短。",
    ),
}


async def main() -> None:
    servers = [uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")) for port, app in APPS.items()]
    for port in APPS:
        print(f"A2A agent 已启动：http://127.0.0.1:{port}/.well-known/agent-card.json")
    print("Ctrl+C 停止。")
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())

