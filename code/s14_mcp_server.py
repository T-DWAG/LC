"""s14_mcp_server.py — 一个最小 MCP 服务：把订单和知识库工具放到独立进程里。

一般不用手动运行，s14_mcp.py 会通过 stdio 把它作为子进程拉起。
单独调试：fastmcp dev code/s14_mcp_server.py （会打开 MCP Inspector 网页）

和 s04 的 @tool 比：
  1. 工具跑在另一个进程，客户端换成别的语言、别的应用（Claude Desktop、Cursor）也能直接用；
  2. 工具名、参数 schema、描述都由协议传过去，客户端不用 import 这份代码；
  3. annotations 是服务端对工具的自我声明（只读 / 有破坏性），客户端可以据此决定要不要人工审批。
数据是本文件里的模拟数据，沿用 s09 / s12；每次拉起子进程都会重置。
"""
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP("refund-desk")

ORDERS = {
    "A100": {"status": "paid", "amount": 199},
    "B200": {"status": "refunded", "amount": 59},
}
KB = {
    "发票": "报销发票抬头为星河科技有限公司，税号 91310115MA1K4QXY2B（演示数据）。",
    "差旅": "差旅住宿标准：一线城市每晚 600 元以内。",
}


@mcp.tool(annotations={"readOnlyHint": True})
def search_kb(query: str) -> str:
    """在公司制度知识库里按关键词检索；没有命中返回「未找到」。"""
    hits = [doc for term, doc in KB.items() if term in query]
    return "\n".join(hits) if hits else "未找到"


@mcp.tool(annotations={"readOnlyHint": True})
def lookup_order(order_id: str) -> dict:
    """按订单号查询订单状态和金额。"""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        # 抛 ToolError → MCP 结果 isError=True → 客户端收到 status="error" 的 ToolMessage，模型能读到原因
        raise ToolError(f"订单 {order_id} 不存在")
    return {"order_id": order_id.strip().upper(), **order}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True})
def refund_order(order_id: str) -> str:
    """对已支付订单执行退款（模拟）。这是写操作。"""
    order = ORDERS.get(order_id.strip().upper())
    if order is None or order["status"] != "paid":
        raise ToolError(f"订单 {order_id} 不可退款")
    order["status"] = "refunded"
    return f"订单 {order_id.strip().upper()} 已退款 {order['amount']} 元（模拟）"


if __name__ == "__main__":
    # stderr 会直接显示在客户端终端里：关掉启动横幅，只打 WARNING 以上的日志。
    mcp.run(show_banner=False, log_level="WARNING")  # 默认 stdio 传输；改成 mcp.run(transport="http", port=8000) 就是远程服务

