"""s14_mcp.py — MCP：Agent 通过协议发现和调用另一个进程里的工具，不再 import 工具代码。

项目根目录运行：python code/s14_mcp.py
它会以 stdio 方式拉起 code/s14_mcp_server.py 作为子进程。自己逐句提问，直接回车退出。
需要模型服务（.env 见 s00_env.md）和 pip install "langchain[mcp]"（langchain>=1.4，自带 langchain.mcp，底层是 FastMCP）。

流程：
  1. MCPAdapter(Path(...)) 按目标类型推断传输：Path → stdio 子进程；"https://..." → Streamable HTTP；
  2. list_tools() 从服务端拿到工具目录，转成普通 LangChain 工具，直接交给 create_agent；
  3. 服务端在 annotations 里声明 refund_order 有破坏性，客户端读元数据，交给 HumanInTheLoopMiddleware 审批，
     不在客户端写死工具名——换一个服务器，规则照样生效（对照 s08 手写 interrupt）。
MCP 工具只有异步实现，所以全程用 ainvoke。
"""
import asyncio
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import SecretStr

warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta")  # langchain.mcp 是 beta，导入时会警告一次
from langchain.mcp import MCPAdapter  # noqa: E402

load_dotenv(override=True)

SERVER = Path(__file__).resolve().parent / "s14_mcp_server.py"

def is_destructive(t: BaseTool) -> bool:
    """读服务端声明的 destructiveHint；每层都可能缺，用 get 给默认值。"""
    info = t.metadata or dict()
    for key in ("mcp", "tool", "annotations"):
        info = info.get(key) or dict()
    return bool(info.get("destructive_hint", False))


model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)


async def main() -> None:
    async with MCPAdapter(SERVER) as adapter:
        tools = await adapter.list_tools()
        destructive = {t.name for t in tools if is_destructive(t)}
        for t in tools:
            print(f"  发现工具 {t.name:<13} {'需审批' if t.name in destructive else '只读'}  {t.description}")

        gate = InterruptOnConfig(allowed_decisions=["approve", "reject"])
        agent = create_agent(
            model,
            tools,
            system_prompt="你是退款客服。查订单、查制度都用工具；退款前先查订单确认已支付。",
            middleware=[HumanInTheLoopMiddleware(interrupt_on={name: gate for name in destructive})],
            checkpointer=InMemorySaver(),  # 审批要暂停再恢复，必须有 checkpointer
        )

        print("\n订单只有 A100（已支付）和 B200（已退款）。直接回车退出。")
        print("试试：「发票抬头是什么」「查一下 A100」「把 A100 退了」「查一下 Z999」")
        config: RunnableConfig = {"configurable": {"thread_id": "mcp-demo"}}
        while True:
            question = input("\n问题：").strip()
            if not question:
                break
            seen = len(agent.get_state(config).values.get("messages", []))  # 只检查本轮新增的消息
            out = await agent.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
            while out.get("__interrupt__"):  # 模型想调有破坏性的工具，暂停等人
                decisions = []
                for req in out["__interrupt__"][0].value["action_requests"]:
                    ok = input(f"  待审批：{req['name']}({req['args']})，批准吗？(y/n)：").strip().lower() == "y"
                    decisions.append({"type": "approve"} if ok else {"type": "reject", "message": "人工拒绝了这次操作"})
                out = await agent.ainvoke(Command(resume={"decisions": decisions}), config)
            for m in out["messages"][seen:]:
                if isinstance(m, ToolMessage) and m.status == "error":
                    print(f"  工具返回错误（已交给模型处理）：{m.text}")
            print("回答：", out["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

