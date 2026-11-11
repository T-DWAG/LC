"""s15_multi_agent.py — 多 Agent 协作：主从（Subagents）、并行（Router）、接力（Handoffs）三种模式。

项目根目录运行：
  python code/s15_multi_agent.py supervisor   主从：主 agent 把专家 agent 当工具调用
  python code/s15_multi_agent.py router       并行：先分类，用 Send 同时发给多个专家，再汇总
  python code/s15_multi_agent.py handoff      接力：前台和退款专员互相转接，谁接手谁直接回复用户
自己逐句提问，直接回车退出。需要模型服务（.env 见 s00_env.md）和 langchain>=1.0。

三个专家沿用前几节的模拟数据：invoice（发票）、travel（差旅）、order（订单查询）。
怎么选（官方 Multi-agent 文档的结论）：
  1. 主从：主 agent 统一调度，子 agent 无记忆、上下文隔离；每次都要多一次主 agent 汇总的调用；
  2. 并行：问题能拆成互不依赖的几块时最快，路由本身无状态，多轮对话要外面再包一层；
  3. 接力：要和用户多轮对话、有先后约束（先核实订单再退款）时用；只能串行，不能并行。
同一进程里的协作用这三种就够了；跨服务、跨团队的 agent 互相调用，用 A2A 协议（见教程）。
"""
import operator
import os
import sys
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)

model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)

# ---------- 三个专家：各自只有自己领域的提示词和工具 ----------
ORDERS = {"A100": "已支付，199 元，下单 5 天", "B200": "已退款，59 元", "C300": "已支付，89 元，下单 45 天"}


@tool
def lookup_order(order_id: str) -> str:
    """按订单号查询订单状态。"""
    print(f"      [order] lookup_order({order_id!r})")
    return ORDERS.get(order_id.strip().upper(), "订单不存在")


EXPERTS = {
    "invoice": create_agent(model, tools=[], system_prompt=(
        "你是发票专员。抬头：星河科技有限公司；税号：91310115MA1K4QXY2B；差旅开普通发票。只回答发票问题，简短。")),
    "travel": create_agent(model, tools=[], system_prompt=(
        "你是差旅专员。住宿：一线城市每晚 600 元，其他 400 元；800 公里以上可坐飞机经济舱。只回答差旅问题，简短。")),
    "order": create_agent(model, tools=[lookup_order], system_prompt=(
        "你是订单专员。用 lookup_order 查订单，只陈述查到的事实，简短。")),
}


def ask_expert(name: str, query: str) -> str:
    """以全新上下文调用专家：专家看不到主对话历史，只看到这一句任务。"""
    print(f"    → 专家 {name} 收到：{query}")
    out = EXPERTS[name].invoke({"messages": [{"role": "user", "content": query}]})
    return out["messages"][-1].content


# ---------- 1. 主从 Subagents：专家包装成工具，主 agent 决定调谁、怎么汇总 ----------
@tool
def ask_invoice(query: str) -> str:
    """咨询发票专员：抬头、税号、发票类型。query 写成一个完整、独立的问题。"""
    return ask_expert("invoice", query)


@tool
def ask_travel(query: str) -> str:
    """咨询差旅专员：住宿标准、交通标准。query 写成一个完整、独立的问题。"""
    return ask_expert("travel", query)


@tool
def ask_order(query: str) -> str:
    """咨询订单专员：订单状态、金额。query 里要带订单号。"""
    return ask_expert("order", query)


supervisor = create_agent(
    model,
    tools=[ask_invoice, ask_travel, ask_order],
    system_prompt="你是行政主管。把问题拆给对应专员（可以同时问多个），再把答案整理成一段回复。不要自己编数据。",
    checkpointer=InMemorySaver(),  # 记忆只在主 agent；专员每次都是新的
)


# ---------- 2. 并行 Router：结构化分类 → Send 扇出 → 汇总 ----------
class Task(BaseModel):
    agent: Literal["invoice", "travel", "order"]
    query: str = Field(description="发给该专员的独立子问题")


class Plan(BaseModel):
    tasks: list[Task] = Field(description="需要咨询的专员；与所有专员无关时为空")


class RouterState(TypedDict):
    question: str
    tasks: NotRequired[list[Task]]
    results: Annotated[list[str], operator.add]  # 多个专家并行写入，用 reducer 合并
    answer: NotRequired[str]


class ExpertInput(TypedDict):
    agent: str
    query: str


planner = model.with_structured_output(Plan)


def classify(state: RouterState) -> dict:
    plan = planner.invoke("把用户问题拆给专员。专员：invoice=发票，travel=差旅，order=订单。\n问题：" + state["question"])
    if not isinstance(plan, Plan):
        raise TypeError(f"分类结果类型不对：{type(plan)}")
    print(f"    分类：{[(t.agent, t.query) for t in plan.tasks]}")
    return {"tasks": plan.tasks}


def fan_out(state: RouterState) -> list[Send] | Literal["synthesize"]:
    if not state["tasks"]:
        return "synthesize"
    # 每个 Send 启动一个 expert 节点实例，它们在同一步里并行执行
    return [Send("expert", {"agent": t.agent, "query": t.query}) for t in state["tasks"]]


def expert(inp: ExpertInput) -> dict:
    return {"results": [f"[{inp['agent']}] {ask_expert(inp['agent'], inp['query'])}"]}


def synthesize(state: RouterState) -> dict:
    if not state["results"]:
        return {"answer": "这个问题不归发票、差旅、订单专员管，请联系 HR。"}
    reply = model.invoke("根据各专员的答复，合成一段回复，不要添加答复里没有的信息。\n问题：" + state["question"]
                           + "\n专员答复：\n" + "\n".join(state["results"]))
    return {"answer": reply.content}


router = (
    StateGraph(RouterState)
    .add_node("classify", classify)
    .add_node("expert", expert)
    .add_node("synthesize", synthesize)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", fan_out, ["expert", "synthesize"])
    .add_edge("expert", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)


# ---------- 3. 接力 Handoffs：平级 agent 用转接工具交出控制权，active_agent 跨轮保留 ----------
class HandoffState(AgentState):
    active_agent: NotRequired[str]


def transfer(to: str, runtime: ToolRuntime) -> Command:
    """通用转接：跳到父图里的另一个 agent 节点，只带「发起转接的 AIMessage + 回执」两条消息。"""
    print(f"    ⇄ 转接给 {to}")
    last_ai = next(m for m in reversed(runtime.state["messages"]) if isinstance(m, AIMessage))
    ack = ToolMessage(content=f"已转接给 {to}", tool_call_id=runtime.tool_call_id)
    return Command(goto=to, update={"active_agent": to, "messages": [last_ai, ack]}, graph=Command.PARENT)


@tool
def transfer_to_refund(runtime: ToolRuntime) -> Command:
    """用户要退款时，转接给退款专员。"""
    return transfer("refund_agent", runtime)


@tool
def transfer_to_front_desk(runtime: ToolRuntime) -> Command:
    """用户的问题与退款无关时，转接回前台。"""
    return transfer("front_desk", runtime)


@tool
def refund_order(order_id: str) -> str:
    """对订单执行退款（模拟）。只有已支付且下单 30 天内的订单可以退。"""
    print(f"      [refund] refund_order({order_id!r})")
    info = ORDERS.get(order_id.strip().upper(), "")
    if not info.startswith("已支付"):
        return "不可退款：订单不存在或未支付"
    if "45 天" in info:
        return "不可退款：超过 30 天退款期"
    return f"订单 {order_id.upper()} 已退款（模拟）"


front_desk_agent = create_agent(model, tools=[ask_invoice, ask_travel, transfer_to_refund], system_prompt=(
    "你是前台。发票、差旅问题用工具咨询专员后回答；用户提到退款，立刻调用 transfer_to_refund，不要自己处理。"))
refund_agent = create_agent(model, tools=[lookup_order, refund_order, transfer_to_front_desk], system_prompt=(
    "你是退款专员，直接和用户对话。缺订单号就问用户；先 lookup_order 核实，再 refund_order。"
    "用户问发票、差旅等非退款问题，调用 transfer_to_front_desk。"))


def route_handoff(state: HandoffState) -> str:
    """最后一条是不带工具调用的 AIMessage → 本轮结束，等用户；否则交给当前 agent。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not last.tool_calls:
        return END
    return state.get("active_agent") or "front_desk"


handoff = (
    StateGraph(HandoffState)
    .add_node("front_desk", front_desk_agent)  # 编译好的 agent 可以直接当子图节点
    .add_node("refund_agent", refund_agent)
    .add_conditional_edges(START, lambda s: s.get("active_agent") or "front_desk", ["front_desk", "refund_agent"])
    .add_conditional_edges("front_desk", route_handoff, ["front_desk", "refund_agent", END])
    .add_conditional_edges("refund_agent", route_handoff, ["front_desk", "refund_agent", END])
    .compile(checkpointer=InMemorySaver())  # 下一轮从上次的 active_agent 继续
)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("supervisor", "router", "handoff"):
        sys.exit("用法：python code/s15_multi_agent.py supervisor|router|handoff")
    print(f"模式：{mode}。订单：A100 已支付、B200 已退款、C300 超 30 天。直接回车退出。")
    print("试试：「去上海出差住多少钱？发票抬头写什么？」「A100 现在什么状态」「我要退款」→「A100」→「发票抬头呢」")
    config: RunnableConfig = {"configurable": {"thread_id": f"{mode}-demo"}}
    while True:
        question = input("\n问题：").strip()
        if not question:
            break
        if mode == "router":
            answer = router.invoke({"question": question, "results": []})["answer"]
        else:
            app = supervisor if mode == "supervisor" else handoff
            answer = app.invoke({"messages": [{"role": "user", "content": question}]}, config)["messages"][-1].content
        if mode == "handoff":
            print(f"    当前接待：{handoff.get_state(config).values.get('active_agent', 'front_desk')}")
        print("回答：", answer)

