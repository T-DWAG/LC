"""将 code/ 中的完整源码和重点行同步到静态教程。

在项目根目录执行：python doc/sync-tutorial-code.py
只使用 Python 标准库；生成的页面可直接通过 file:// 打开。
"""
from __future__ import annotations

import ast
import html
import io
import keyword
from pathlib import Path
import re
import sys
import token
import tokenize

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "doc" / "langgraph-zero2hero.html"
PAGE_V2 = ROOT / "doc" / "langgraph-zero2hero-v2.html"
# 与每节图示对应的重点行；其余代码仍完整显示。
HIGHLIGHTS = {
    "s01_probe.py": [r"^def make_llm", r"^    return ChatOpenAI", r"^    llm = make_llm", r"^    msg=llm\.invoke", r"^    print\(msg\.content"],
    "s02_prompt_model.py": [r"^prompt = ChatPromptTemplate", r"^    \(\"(?:system|user)\"", r"^chain = prompt", r"^response = chain\.invoke", r"^print\(response\.content"],
    "s03_structure.py": [r"^class Brief", r"^    (?:title|points):", r"^llm=make_llm\(\)\.with_structured_output", r"^result=llm\.invoke", r"^if not isinstance\(result, Brief\)"],
    "s04_tool.py": [r"^@tool", r"^def (?:add|multiply)\(", r"^agent = create_react_agent", r"^result =agent\.invoke", r'"messages": \[\{"role"'],
    "s05_min_graph.py": [r"^class State\(", r"^    (?:topic|joke|polished):", r"^def (?:generate_topic|write_joke|polish)\(", r"^graph = StateGraph", r"^graph\.add_(?:node|edge)", r"^result = graph\.compile"],
    "s06_calc_agent.py": [r"^model_with_tools =", r"^class MessageState", r"^    messages:Annotated", r"^    llm_calls:int", r"^def llm_call", r"^agent_builder =", r"^agent_builder\.add_(?:node|edge|conditional_edges)", r"^agent = agent_builder\.compile"],
    "s07_checkpoint.py": [r"^def call_model", r"^    msgs=", r"^    return \{\"messages\"", r"^    StateGraph\(MessagesState\)", r"^    \.add_(?:node|edge)", r"^    \.compile\(checkpointer", r"^cfg: RunnableConfig", r"^r[12]=app\.invoke"],
    "s09_refund_agent.py": [r"^class RefundRequest", r"^class RefundState", r"^    clarify_turns:", r"^CLARIFY_TURNS", r"^@tool", r"^def lookup_order", r"^extractor =", r"^def (?:extract_request|route_request|clarify|route_clarify|check_order|check_policy|route_policy|approve_refund|refund|reject)\(", r"^    (?:decision|reply) = interrupt", r"^    return Command\(goto", r"^builder\.(?:add_node|add_edge|add_conditional_edges)", r"^app = builder\.compile", r"^        result = app\.invoke", r"^        if payload\.get\(\"kind\"\) == \"clarify\":"],
    "s08_hitl.py": [r"^def approval_node", r"^    decision = interrupt", r"^    return Command\(goto", r"^    StateGraph\(ApprovalState\)", r"^    \.add_(?:node|edge)", r"^    \.compile\(checkpointer", r"^paused = graph\.invoke", r"^approved = graph\.invoke", r"^rejected = graph\.invoke"],
}
# 第二版页面里由本脚本同步的节（s10–s12 的代码区仍保持手工维护）。
HIGHLIGHTS_V2 = {
    "s13_skills.py": [r"^def parse_skill", r"^SKILLS = ", r"^@tool", r"^def load_skill", r"^class SkillMiddleware", r"^    tools = \[load_skill\]", r"^    def wrap_model_call", r"^        return handler\(request\.override", r"^    return create_agent"],
    "s13_skills_deepagents.py": [r"^backend = FilesystemBackend", r"^SOURCES = ", r"^                FilesystemMiddleware\(backend", r"^                SkillsMiddleware\(backend", r"^    return create_deep_agent", r"^        skills=SOURCES", r"^        permissions="],
    "s14_mcp_server.py": [r"^mcp = FastMCP", r"^@mcp\.tool", r"^        raise ToolError", r"^    mcp\.run"],
    "s14_mcp.py": [r"^from langchain\.mcp import MCPAdapter", r"^def is_destructive", r"^    async with MCPAdapter", r"^        tools = await adapter\.list_tools", r"^        destructive = ", r"^        gate = InterruptOnConfig", r"^            middleware=\[HumanInTheLoopMiddleware", r"^            out = await agent\.ainvoke", r"^                out = await agent\.ainvoke\(Command"],
    "s15_multi_agent.py": [r"^def ask_expert", r"^supervisor = create_agent", r"^class Plan", r"^    results: Annotated", r"^def fan_out", r"^    return \[Send", r"^def transfer", r"^    return Command\(goto=to", r"^def route_handoff", r"^handoff = \("],
    "s16_a2a_agents.py": [r"^class LangChainExecutor", r"^    async def execute", r"^        user = context\.metadata", r"^            await updater\.reject", r"^        config: RunnableConfig = ", r"^        await updater\.complete", r"^class BearerAuth", r"^    card = AgentCard"],
    "s16_a2a.py": [r"^async def send", r"^        metadata=", r"^    async for response in client\.send_message", r"^def make_tool", r"^        description=f", r"^                card = await A2ACardResolver", r"^            tools\.append"],
}


def render_source(source: str, patterns: list[str]) -> str:
    """按 token 着色，按源文件行匹配强调；保留所有源代码字符。"""
    lines = source.splitlines(keepends=True)
    decorations: list[list[tuple[int, int, str]]] = [[] for _ in lines]
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        kind = None
        if tok.type == token.COMMENT:
            kind = "comment"
        elif tok.type == token.STRING:
            kind = "string"
        elif tok.type == token.NUMBER:
            kind = "number"
        elif tok.type == token.NAME and keyword.iskeyword(tok.string):
            kind = "keyword"
        elif tok.type == token.OP and tok.string == "@":
            kind = "keyword"
        if not kind:
            continue
        start_row, start_col = tok.start
        end_row, end_col = tok.end
        for row in range(start_row, end_row + 1):
            if row > len(lines):
                continue
            line_length = len(lines[row - 1].rstrip("\r\n"))
            start = start_col if row == start_row else 0
            end = min(end_col if row == end_row else line_length, line_length)
            if start < end:
                decorations[row - 1].append((start, end, kind))
    result = []
    for line, ranges in zip(lines, decorations):
        text = line.rstrip("\r\n")
        rendered = []
        pos = 0
        for start, end, kind in sorted(ranges):
            rendered.append(html.escape(text[pos:start]))
            rendered.append(f'<span class="tok-{kind}">{html.escape(text[start:end])}</span>')
            pos = end
        rendered.append(html.escape(text[pos:]))
        content = "".join(rendered)
        if any(re.search(pattern, text) for pattern in patterns):
            content = f'<span class="focus-line">{content}</span>'
        # read_text 使用通用换行；换行在 span 外，复制时保留原始代码结构。
        result.append(content + line[len(text):])
    return "".join(result)


def sync_guide(page: str) -> str:
    source = (ROOT / "code" / "s09_refund_agent.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    def excerpt(name: str) -> str:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return "".join(lines[start - 1:node.end_lineno]).rstrip()

    def between(start: str, end: str) -> str:
        begin = next(i for i, line in enumerate(lines) if line.startswith(start))
        finish = next(i for i in range(begin, len(lines)) if lines[i].startswith(end))
        return "".join(lines[begin:finish + 1]).rstrip()

    steps = [
        ("01 · 先确定数据", "哪些信息需要在节点之间传递？把用户原文、提取结果、追问次数、订单和最终结果放进共享状态。NotRequired 表示字段可以稍后才写入。", excerpt("RefundState")),
        ("02 · 约束模型输出", "模型只负责从用户原文中提取订单号和原因；这两个字段是模型的输出契约，不是整个图的状态。", excerpt("RefundRequest") + "\n\n" + 'extractor = llm.with_structured_output(RefundRequest)'),
        ("03 · 实现提取节点", "节点接收当前状态，调用模型，再返回需要写回状态的字段。空字符串当成没提供。", excerpt("extract_request")),
        ("04 · 缺信息就追问三步", "route_request 发现订单号或原因缺失时进入 clarify，不直接结束。clarify 用 interrupt 暂停，把用户补充合并进已有字段。未满 3 步回到 clarify；满 3 步后齐了去 lookup，仍缺才写入 missing。", excerpt("route_request") + "\n\n" + excerpt("clarify") + "\n\n" + excerpt("route_clarify") + "\n\n" + excerpt("missing_info")),
        ("05 · 查订单并检查规则", "工具查本地订单，节点写入状态；确定性代码判断订单是否存在、已支付且未超过 30 天。模型不决定能否退款。", excerpt("lookup_order") + "\n\n" + excerpt("check_order") + "\n\n" + excerpt("check_policy") + "\n\n" + excerpt("route_policy")),
        ("06 · 设计人工介入", "对符合规则的订单暂停；恢复时校验布尔值，再用 Command 决定执行模拟退款还是拒绝。批准不代表执行成功。", excerpt("approve_refund") + "\n\n" + excerpt("refund") + "\n\n" + excerpt("reject")),
        ("07 · 将节点连成图", "先注册节点，再建立入口、条件路由和结束边。extract 按字段是否齐全去 lookup 或 clarify；clarify 自己决定继续追问、查单或结束。approval 通过 Command 动态跳转，所以这里不添加它到 refund / reject 的普通边。", between("builder = StateGraph", "app = builder.compile")),
        ("08 · 调用、暂停和恢复", "同一个线程上循环 invoke。中断 kind 为 clarify 时，把补充文字作为 resume 传回；kind 为 approval 时，把 y/n 变成布尔值。节点恢复时会从头运行到 interrupt，中断之前的副作用必须可安全重试。", between('if __name__ == "__main__":', '    print("结果："')),
    ]
    output = ['<div class="coder-guide"><h3>从程序员的视角：边设计边对照代码</h3><p>先看左边要解决的问题，再看右边源码。以下片段按设计顺序排列，来自同一个 s09 文件，并非独立可运行程序。</p>']
    for title, explanation, code in steps:
        output.append(f'<div class="guide-step"><div><h4>{html.escape(title)}</h4><p>{html.escape(explanation)}</p></div><pre><code>{render_source(code, [])}</code></pre></div>')
    output.append('<p><strong>最后验证：</strong>只输入「我要退款」，应连续追问 3 步，补上 A100 和原因后才进入审批。A100 分别批准和拒绝；B200 已退款，C300 超期，D400 模拟退款失败；另测未知订单。模型可能提取错误，观察状态及实际路径。</p></div>')
    page, count = re.subn(r'<!-- guide:start -->.*?<!-- guide:end -->', lambda _: '<!-- guide:start -->' + ''.join(output) + '<!-- guide:end -->', page, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("找不到 09 导读区域")
    return page


def sync_blocks(page: str, highlights: dict[str, list[str]]) -> str:
    for name, patterns in highlights.items():
        source = (ROOT / "code" / name).read_text(encoding="utf-8")
        block = (f'<div class="code-label">完整代码 · {name} <span>绿色底色为重点</span></div>'
                        f'<pre class="source-code"><code>{render_source(source, patterns)}</code></pre>')
        # 每一节只替换自己的代码区，允许再次执行同步。
        pattern = rf'<div class="code-label">[^<]*{re.escape(name)}(?:\s*<span>.*?</span>)?</div><pre(?: class="source-code")?>(?:<code>)?.*?(?:</code>)?</pre>'
        page, count = re.subn(pattern, lambda _: block, page, count=0, flags=re.DOTALL)
        if count != 1:
            raise RuntimeError(f"找不到或重复的代码区域：{name} ({count})")
    return page


def main() -> None:
    page = sync_guide(sync_blocks(PAGE.read_text(encoding="utf-8"), HIGHLIGHTS))
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"已同步 {len(HIGHLIGHTS)} 个脚本到 {PAGE}")
    page = sync_blocks(PAGE_V2.read_text(encoding="utf-8"), HIGHLIGHTS_V2)
    sys.path.insert(0, str(ROOT / "doc"))
    import design_notes, design_render  # noqa: E401  s14-s16 的工程设计讲解
    page = design_render.sync(page, ROOT, design_notes.DESIGNS, render_source)
    PAGE_V2.write_text(page, encoding="utf-8", newline="\n")
    print(f"已同步 {len(HIGHLIGHTS_V2)} 个脚本到 {PAGE_V2}")


if __name__ == "__main__":
    main()
