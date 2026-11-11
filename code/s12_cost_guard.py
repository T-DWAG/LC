"""s12_cost_guard.py — 成本护栏：用 LangChain 自带的中间件给 Agent 设上限，不自己造计数器。

项目根目录运行：python code/s12_cost_guard.py
自己逐句提问，直接回车退出。知识库只有「发票」「差旅」两条（模拟）。
需要模型服务（.env 见 s00_env.md）和 langchain>=1.0。

Agent 的成本来自「模型调用次数 × 每次的 token」，失控通常是循环：
  1. 查不到资料，模型换个词一直查 → ToolCallLimitMiddleware 限制工具次数；
  2. 单次提问里模型反复思考 → ModelCallLimitMiddleware(run_limit) 限制本轮调用；
  3. 同一会话里用户一直追问 → ModelCallLimitMiddleware(thread_limit) 限制整个会话，需要 checkpointer；
  4. 花了多少 token → UsageMetadataCallbackHandler 按模型统计真实用量。
官方没有「按 token 封顶」的中间件；按金额/租户的硬配额要放在网关或计费服务里。
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

load_dotenv(override=True)

KB = {
    "发票": "报销发票抬头为星河科技有限公司，税号 91310115MA1K4QXY2B（演示数据）。",
    "差旅": "差旅住宿标准：一线城市每晚 600 元以内。",
}


@tool
def search_kb(query: str) -> str:
    """在公司制度知识库里按关键词检索，返回命中的条目；没有命中返回「未找到」。"""
    print(f"      工具调用：search_kb({query!r})")
    hits = [doc for term, doc in KB.items() if term in query]
    return "\n".join(hits) if hits else "未找到"


model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)

agent = create_agent(
    model,
    tools=[search_kb],
    system_prompt="你是公司制度助手。只根据 search_kb 的结果回答；查不到就说知识库没有，请联系 HR。",
    middleware=[
        # 一次提问最多查 2 次库；超了就告诉模型「别再查了」，让它基于已有结果收尾。
        ToolCallLimitMiddleware(tool_name="search_kb", run_limit=2, exit_behavior="continue"),
        # 一次提问最多 4 次模型调用；整个会话最多 8 次。到上限直接结束并回一条提示，不再调模型。
        ModelCallLimitMiddleware(run_limit=4, thread_limit=8, exit_behavior="end"),
    ],
    checkpointer=InMemorySaver(),  # thread_limit 要跨多次 invoke 累计，必须有 checkpointer
)


if __name__ == "__main__":
    print("知识库只有「发票」「差旅」。单次提问最多查库 2 次、调模型 4 次；整个会话最多调模型 8 次。直接回车退出。")
    config: RunnableConfig = {"configurable": {"thread_id": "cost-demo"}}
    usage = UsageMetadataCallbackHandler()  # 跨轮累计，按模型名统计 input/output/total tokens
    while True:
        question = input("\n问题：").strip()
        if not question:
            break
        before = agent.get_state(config).values.get("thread_model_call_count", 0)
        out = agent.invoke({"messages": [{"role": "user", "content": question}]}, {**config, "callbacks": [usage]})
        print("回答：", out["messages"][-1].content)
        # 计数由中间件写进图状态（私有字段，不在 invoke 返回值里），从检查点读。
        counts = agent.get_state(config).values
        used = counts.get("thread_model_call_count", 0)
        tools = counts.get("thread_tool_call_count", {}).get("search_kb", 0)
        total = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        print(f"计数  本轮调模型 {used - before}/4，会话调模型 {used}/8，会话查库 {tools} 次，会话累计 {total} tokens")
