"""s14–s16 的「工程设计」讲解内容，由 sync-tutorial-code.py 渲染进 v2 页面。

文字里用 `反引号` 表示行内代码。代码片段不手抄，按名字或行首前缀从 code/ 源码截取，
源码改了重新同步即可。

代码片段写法：
  "name"  顶层函数 / 类（含装饰器）；不是标识符的字符串表示「以它开头的那一行」
  ("lines", "开头前缀", "结尾前缀")  从第一行以「开头前缀」开始（忽略缩进），到之后第一行以「结尾前缀」开始为止，含两端
"""

DESIGNS = {
    # ------------------------------------------------------------------ s14
    "s14": {
        "intro": (
            "s14 把 s04 里和 Agent 写在一起的工具拆成了两个进程：服务端负责「有哪些工具、怎么执行」，"
            "客户端负责「怎么用这些工具」。两边只通过 MCP 协议传消息，客户端不 import 服务端的任何代码。"
            "读代码时可以按两个问题来看：工具契约由哪一方定义，审批规则又由哪一方定。"
        ),
        "files": [
            ("`s14_mcp_server.py`", "服务端 · 工具的所有者",
      "持有数据（`ORDERS`、`KB`）；用 `@mcp.tool` 声明工具，函数签名生成参数 schema，docstring 生成描述；"
      "用 `annotations` 声明只读或有破坏性；业务错误抛 `ToolError`。",
      "不知道调用它的是哪个 Agent、哪个模型"),
            ("`s14_mcp.py`", "客户端 · 工具的使用者",
      "拉起并连接服务端；运行时拿到工具目录；把注解翻译成审批策略；组装 Agent；跑对话循环，处理暂停和恢复。",
      "只依赖 MCP 协议和工具元数据，不依赖服务端源码"),
        ],
        "sequence": [
            "`async with MCPAdapter(SERVER)`：参数是 `Path`，按 stdio 传输处理，拉起 `python s14_mcp_server.py` 子进程，完成 MCP 握手。stdout 被协议占用，服务端的日志只能走 stderr。",
            "`adapter.list_tools()`：发送 `tools/list`，服务端返回每个工具的名字、描述、JSON Schema 和 annotations；适配器把它们转成 LangChain 的 `BaseTool`，元数据放在 `t.metadata`。",
            "`is_destructive` 逐层读取 `metadata → mcp → tool → annotations → destructive_hint`，得到需要审批的工具名集合 `destructive`。",
            "`create_agent` 组装 Agent：模型 + 工具 + `HumanInTheLoopMiddleware(interrupt_on=...)` + `InMemorySaver`。审批中间件只拦 `destructive` 里的工具。",
            "每轮先记下已有消息数 `seen`，再 `ainvoke`。模型想调 `refund_order` 时，中间件在执行前 `interrupt`，结果里出现 `__interrupt__`。",
            "客户端逐个询问 `action_requests`，按相同顺序组装 `decisions`，用 `Command(resume={\"decisions\": ...})` 在同一个 `thread_id` 上恢复。批准就真正发出 `tools/call`，拒绝就把拒绝理由作为工具结果交给模型。",
            "服务端抛出的 `ToolError` 变成 `status=\"error\"` 的 `ToolMessage`。客户端只打印 `seen` 之后新增的错误，模型自己根据错误原因组织回答。",
            "退出 `async with` 时关闭会话并结束子进程，服务端内存里的订单数据随之清空。",
        ],
        "decisions": [
            ("审批规则读服务端注解，不写死工具名",
      "工具由服务端维护。换一个服务端、或者服务端新增了写操作，客户端不用改代码，规则照样生效。",
      "注解只是服务端的自我声明。服务端写错或者被篡改，客户端就会放行，所以服务端自己还要校验。"),
            ("`refund_order` 在服务端再查一次状态",
      "两道防线：客户端审批管「人同不同意」，服务端校验管「这件事能不能做」。审批通过不等于业务上允许。",
      "规则有两份；业务规则以服务端为准。"),
            ("业务失败抛 `ToolError`，不返回普通字符串",
      "MCP 结果带 `isError` 标记，客户端能把失败和正常结果区分开，可以统计、告警，或者只打印错误。",
      "传输层错误（子进程崩溃、连接断开）不会变成 ToolMessage，会直接抛到 `ainvoke` 外面。"),
            ("全程 `async` / `ainvoke`",
      "适配器生成的 MCP 工具只有异步实现，同步 `invoke` 调不了。",
      "入口要 `asyncio.run`，`input()` 会阻塞事件循环；在演示里可以接受。"),
            ("checkpointer 放在客户端 Agent 上",
      "`interrupt` 暂停后要从检查点恢复，和 s08、s10 是同一个机制。",
      "示例用 `InMemorySaver`，进程退出后审批状态就丢了；生产环境换成 s10 的持久化存储。"),
            ("数据放在服务端进程内存",
      "演示时每次拉起子进程都会重置，方便反复跑。",
      "生产环境放数据库；如果是多个客户端共享的 HTTP 服务，还要加鉴权。"),
        ],
        "steps": [
            ("01 · 服务端：声明工具契约", "s14_mcp_server.py",
      "`FastMCP(\"refund-desk\")` 是服务名。`@mcp.tool` 把函数变成协议里的工具：参数 schema 来自类型注解，描述来自 docstring。"
      "`readOnlyHint` 告诉客户端这个工具不改数据。和 s04 的 `@tool` 相比，写法几乎一样，变的只是运行位置和调用方。",
      ["mcp = FastMCP", "search_kb", "lookup_order"]),
            ("02 · 服务端：写操作自己守门", "s14_mcp_server.py",
      "`destructiveHint=True` 是给客户端看的提示，真正的保护在函数体里：订单不存在或者不是 paid 就抛 `ToolError`，不执行退款。"
      "`mcp.run()` 默认走 stdio；改成 `transport=\"http\"` 就是远程服务，工具代码不用动。",
      ["refund_order", ("lines", 'if __name__ == "__main__":', "mcp.run(")]),
            ("03 · 客户端：连接与发现", "s14_mcp.py",
      "`MCPAdapter` 按参数类型选择传输方式，`Path` 对应 stdio 子进程。`list_tools()` 返回的是普通 LangChain 工具，"
      "后面的 `create_agent` 不需要知道它们来自 MCP。所有工作都在 `async with` 里完成，连接在块结束时关闭。",
      [("lines", "async with MCPAdapter", 'print(f"  发现工具')]),
            ("04 · 客户端：把注解翻译成策略", "s14_mcp.py",
      "`is_destructive` 是客户端唯一需要了解 MCP 元数据结构的地方，每一层都用 `get` 给默认值，缺了就按只读处理。"
      "得到工具名集合后生成 `interrupt_on`，交给官方的审批中间件，客户端不用自己写 interrupt 节点。",
      ["is_destructive", ("lines", "gate = InterruptOnConfig", ")")]),
            ("05 · 客户端：对话循环里的暂停与恢复", "s14_mcp.py",
      "一个 `__interrupt__` 里可能有多个 `action_requests`，因为模型可能一次要求调用多个写工具。`decisions` 的顺序必须和它们一致。"
      "用 `while` 而不是 `if`：恢复后模型还可能接着调用下一个写工具。`seen` 用来只检查本轮新增的消息。",
      [("lines", "while True:", 'print("回答："')]),
        ],
    },
    # ------------------------------------------------------------------ s15
    "s15": {
        "intro": (
            "s15 只有一个文件，分三层：共享层定义一次专家，三种编排层各自组装一个可调用对象，入口层按命令行参数选一个运行。"
            "三种模式用的是同一组专家，区别只在编排：谁来决定调用哪个专家、状态存在哪里、谁来回复用户。"
            "读代码时可以对比三种模式各自的 State 是什么、控制权在谁手里。"
        ),
        "files": [
            ("共享层", "`model` · `ORDERS` · `lookup_order` · `EXPERTS` · `ask_expert`",
      "三个专家各有自己的提示词和工具；`ask_expert` 是调用专家的唯一入口，每次传入全新的 `messages`，专家看不到主对话。",
      "三种模式共用，专家代码不会因为编排方式而改"),
            ("编排 1 · 主从", "`ask_invoice` / `ask_travel` / `ask_order` · `supervisor`",
      "专家包成工具，主 agent 在 ReAct 循环里自己决定调谁、调几次，最后汇总。",
      "状态是主 agent 的 `messages`，检查点只挂在主 agent 上"),
            ("编排 2 · 并行", "`Task` · `Plan` · `RouterState` · `ExpertInput` · `classify` / `fan_out` / `expert` / `synthesize` · `router`",
      "一张显式的 StateGraph：先分类拆任务，再用 `Send` 扇出，最后汇总。分类只做一次，之后不再让模型决定流程。",
      "状态是 `RouterState`，无检查点，每次问答都是独立的"),
            ("编排 3 · 接力", "`HandoffState` · `transfer` · `transfer_to_*` · `refund_order` · 两个 agent · `route_handoff` · `handoff`",
      "两个平级 agent 作为父图的子图节点，转接工具通过 `Command(graph=PARENT)` 改变父图的走向。",
      "状态是父图的 `messages + active_agent`，检查点在父图上"),
            ("入口层", "`if __name__ == \"__main__\"`",
      "按 `mode` 选对象：router 的输入是 `question`，另外两个的输入是 `messages`；handoff 每轮打印当前接待人。",
      "三个对象的输入和输出形状不同，入口层负责适配"),
        ],
        "compare": {
            "head": ["", "主从 supervisor", "并行 router", "接力 handoff"],
            "rows": [
  ["谁决定调用谁", "主 agent，每一步都可能改主意", "`classify` 一次性决定", "当前 agent 调转接工具"],
  ["专家看到的上下文", "一句 query", "一句 query", "父图的完整 `messages`"],
                ["谁回复用户", "主 agent 汇总", "`synthesize` 汇总", "当前接手的 agent 直接回复"],
                ["一次问 N 个专家的模型调用", "主 agent ≥2 次 + 每个专家各自的调用", "1 次分类 + N 个专家 + 1 次汇总，次数固定", "串行，只有当前 agent 在调用"],
  ["多轮记忆", "主 agent 的 checkpointer", "没有，需要外面再包一层", "父图的 checkpointer + `active_agent`"],
                ["适合", "问题开放、需要边做边判断", "可拆成互不依赖的子问题，追求延迟和成本可预测", "需要多轮对话，并且有阶段（先核实再退款）"],
            ],
        },
        "sequence": [
            "**主从**：用户问题 → 主 agent 的模型一次发出 `ask_travel` 和 `ask_invoice` 两个工具调用 → 每个工具里 `ask_expert` 用新上下文调用专家 → 两个 ToolMessage 回到主 agent → 主 agent 再调用一次模型，汇总回答。",
            "**并行**：`START → classify`，结构化输出得到 `Plan.tasks` → `fan_out` 为每个 task 返回一个 `Send(\"expert\", {...})` → 同一步里并发执行多个 `expert` 实例，每个都向 `results` 追加一条 → reducer `operator.add` 合并 → `synthesize` 汇总 → `END`。`tasks` 为空时 `fan_out` 直接返回 `\"synthesize\"`，由它给出转 HR 的兜底回答。",
            "**接力**：第一轮 `START` 读 `active_agent`（为空）进入 `front_desk` → 用户说要退款，前台调用 `transfer_to_refund` → 工具返回 `Command(goto=\"refund_agent\", graph=PARENT)`，把 `active_agent` 写进父图 → `refund_agent` 向用户追问订单号，最后一条是不带工具调用的 AIMessage，`route_handoff` 返回 `END`，本轮结束。第二轮用户回复「A100」，`START` 读到 `active_agent=refund_agent`，直接进入退款专员。",
        ],
        "decisions": [
            ("专家只定义一次，所有调用都经过 `ask_expert`",
      "对比三种模式时，变量只有编排方式；上下文隔离和日志也只需要在一个地方处理。",
      "专家没有记忆。如果需要专家记住上下文，要在 `ask_expert` 里为每个会话传入 thread_id。"),
            ("主从：工具 docstring 就是选人依据",
      "主 agent 只能看到工具名和描述，所以描述要写清专家管什么、query 要怎么写（完整、独立的问题）。",
      "描述写得不清楚就会选错专家，问题和 s13 的技能描述一样。"),
            ("并行：`Task.agent` 用 `Literal` 约束",
      "结构化输出只能从三个专家名里选，后面 `EXPERTS[name]` 不会因为模型编造了一个名字而 KeyError。",
      "新增专家要同时改 `Literal`、`EXPERTS` 和分类提示词。"),
            ("并行：`ExpertInput` 和 `RouterState` 分开",
      "`Send` 发给 expert 节点的是私有输入，只有 agent 和 query，节点不需要看全局状态；它的返回值仍写回 `RouterState`。",
      "要多定义一个类型，但每个 expert 实例的输入范围更清楚。"),
            ("并行：`results` 必须带 reducer",
      "多个 expert 在同一步写同一个键。没有 `operator.add` 时，LangGraph 不知道怎么合并，会报 `InvalidUpdateError`。",
      "合并顺序不保证，所以 `synthesize` 只汇总，不依赖顺序。"),
            ("接力：转接工具只带回两条消息",
      "`Command.PARENT` 跳出子图时，子图内部的消息不会自动带回父图。带上发起转接的 AIMessage 和对应的 ToolMessage 回执，父图里的工具调用才能成对，下一个 agent 调用模型时不会报错。",
      "子图里的其他中间消息不会留在父图。"),
            ("接力：`active_agent` 存在父图状态里",
      "`START` 的条件边和 `route_handoff` 都读取它，加上父图的 checkpointer，下一轮会直接回到上次接待的人，实现粘性会话。",
      "两个 agent 共享完整的对话历史；如果需要隔离，就要在转接时裁剪消息。"),
        ],
        "steps": [
            ("01 · 共享层：专家只定义一次", "s15_multi_agent.py",
      "`EXPERTS` 是一个名字到 agent 的映射，三种模式都通过名字取专家。`ask_expert` 每次都传入只含一条用户消息的新列表，这就是上下文隔离：专家看不到主对话，也不会被无关历史干扰。",
      ["lookup_order", ("lines", "EXPERTS = {", "}"), "ask_expert"]),
            ("02 · 主从：专家变成工具", "s15_multi_agent.py",
      "每个专家包成一个 `@tool`，函数体只有一行。工具名和 docstring 是主 agent 选人的依据。`checkpointer` 只挂在 `supervisor` 上，多轮记忆在主 agent。",
      ["ask_invoice", ("lines", "supervisor = create_agent", ")")]),
            ("03 · 并行：先设计数据", "s15_multi_agent.py",
      "`Task` / `Plan` 是模型的输出契约；`RouterState` 是图的共享状态；`ExpertInput` 是 `Send` 发给 expert 节点的私有输入。三个类型分别对应「模型说了什么」「图里存什么」和「每个并发分支拿到什么」。",
      ["Task", "Plan", "RouterState", "ExpertInput"]),
            ("04 · 并行：分类、扇出、汇总", "s15_multi_agent.py",
      "`classify` 唯一一次调用模型做决定，返回类型不对就直接报错，不往下猜。`fan_out` 是条件边函数，可以返回节点名，也可以返回 `Send` 列表。`expert` 只返回 `results` 增量，`synthesize` 处理空结果的兜底。",
      [("lines", "planner = ", "planner = "), "classify", "fan_out", "expert", "synthesize"]),
            ("05 · 并行：连图", "s15_multi_agent.py",
      "`add_conditional_edges(\"classify\", fan_out, [\"expert\", \"synthesize\"])` 的第三个参数列出所有可能的目标，用来校验和画图。`expert → synthesize` 是普通边，等这一步所有并发的 expert 都结束后，`synthesize` 才运行一次。",
      [("lines", "router = (", ")")]),
            ("06 · 接力：转接就是控制流", "s15_multi_agent.py",
      "`HandoffState` 在 agent 默认状态上加了 `active_agent`。`transfer` 通过 `ToolRuntime` 拿到当前状态和 `tool_call_id`，构造回执，返回一个作用于父图的 `Command`：既跳转，又更新 `active_agent` 和 `messages`。",
      ["HandoffState", "transfer", "transfer_to_refund"]),
            ("07 · 接力：路由与粘性会话", "s15_multi_agent.py",
      "编译好的 agent 直接作为父图节点。`route_handoff` 的规则是：最后一条是没有工具调用的 AIMessage，就结束本轮等用户；否则交给 `active_agent`。转接时它读到的已经是新的 `active_agent`，和 `goto` 指向同一个节点，两者不冲突。父图的 checkpointer 让下一轮从 `START` 直接回到上次的接待人。",
      ["route_handoff", ("lines", "handoff = (", ")")]),
        ],
    },
    # ------------------------------------------------------------------ s16
    "s16": {
        "intro": (
            "s16 把 s15 主从模式里的「专家」换成另一个团队部署的服务。服务端分成三层：协议层（A2A SDK 提供）、适配层（`LangChainExecutor`，唯一需要自己写的协议代码）、"
            "业务层（`create_agent`）；鉴权是一个 HTTP 中间件。编排方分成四步：发现 → 适配 → 传输 → 编排。"
            "和 s15 对照着看，`ask_expert` 的位置换成了 `ask_remote`，其余结构基本不变。"
        ),
        "files": [
            ("`s16_a2a_agents.py` 协议层", "`create_agent_card_routes` · `create_jsonrpc_routes` · `DefaultRequestHandler` · `InMemoryTaskStore`",
      "SDK 提供：公开 Agent Card，解析 JSON-RPC 的 SendMessage，管理 task 的存储和状态流转，把事件变成响应。",
      "不需要自己写"),
            ("`s16_a2a_agents.py` 适配层", "`LangChainExecutor`",
      "把一条 A2A 消息翻译成一次 agent 调用：登记 task → 检查身份 → WORKING → `ainvoke` → 写 artifact → COMPLETED。",
      "换框架只改这一层，调用方完全无感"),
            ("`s16_a2a_agents.py` 业务层与工厂", "`build_app` · `APPS` · `main`",
      "`build_app` 用一组参数（端口、名字、描述、维护人、技能、提示词）生成一个完整的服务；`APPS` 模拟两个团队；`main` 在一个进程里同时运行两个 uvicorn。",
      "两个服务在同一进程里只是为了方便演示，彼此不共享状态"),
            ("`s16_a2a_agents.py` 横切", "`BearerAuth`",
      "POST 请求没有正确的令牌就返回 401；GET Agent Card 公开，保证能被发现。",
      "回答「调用方是谁」，不关心替谁调用"),
            ("`s16_a2a.py` 编排方", "`REGISTRY` · `make_tool` · `send` · `main`",
      "读注册表 → 拉 Agent Card → 每个远程 agent 生成一个工具（闭包里持有 client 和 context_id）→ 交给主 agent。",
      "只依赖对方的 Agent Card 和协议，不知道对方内部实现"),
        ],
        "sequence": [
            "**启动**：服务端监听 9101、9102。编排方对 `REGISTRY` 里每个地址 `GET /.well-known/agent-card.json`，拿到名字、描述、技能、版本、维护人和接口地址。",
            "**适配**：`make_tool(card, client)` 为每个远程 agent 生成一个 `StructuredTool`。工具名来自 `card.name`（`-` 换成 `_`），描述由 `card.description` 和 skills 拼成；同时生成一个 `context_id`，存在闭包里。",
            "**自检**：用不带令牌的 httpx 客户端发一次，预期 `BearerAuth` 返回 401，客户端抛 `A2AClientError`。",
            "**一次提问**：主 agent 的模型同时选中 `travel_agent` 和 `invoice_agent` → 每个工具调用 `send`：构造 `SendMessageRequest`，`message.context_id` 用闭包里的 id，`metadata.user` 填真实用户 → POST JSON-RPC。",
            "**服务端处理**：`BearerAuth` 放行 → `DefaultRequestHandler` 调用 `execute` → 新建 task 并入队（SUBMITTED）→ `user` 不在白名单就 `reject`（REJECTED，附原因）并返回 → 否则 `start_work`（WORKING）→ 用 `context_id` 作为 `thread_id` 调用 agent → `add_artifact` → `complete`（COMPLETED）。execute 里抛出的异常由框架转成 FAILED。",
            "**取结果**：`streaming=False` 时 `send_message` 只返回最终 task。COMPLETED 就取 artifacts 里的文字；其他状态返回「未完成（状态）：原因」。外面的 `wait_for(60)` 兜住超时，任何失败都变成工具结果的文字，主 agent 照常汇总。",
            "**追问**：用户再问「那如果去成都呢」，主 agent 再次调用 `travel_agent`，带的是同一个 `context_id`，服务端的 checkpointer 里有上一轮记录，所以能接上「出差住宿」的上下文。",
        ],
        "decisions": [
            ("只自己写 `AgentExecutor`",
      "协议细节（JSON-RPC、task 存储、状态机、Agent Card 路由）都交给 SDK，自己只写业务和协议之间的翻译。",
      "`cancel` 没有实现；长任务要支持取消，需要在 execute 里配合检查。"),
            ("鉴权和授权分两层",
      "令牌在 HTTP 中间件里检查（调用方是谁，失败返回 401，请求进不了 agent）；用户权限在 execute 里检查（替谁调用，失败返回 REJECTED 和原因，主 agent 能如实转告）。",
      "演示里 `metadata.user` 可以随便填。生产环境要由网关签名或换成用户令牌，否则谁都能冒充 alice。"),
            ("Agent Card 公开，SendMessage 需要令牌",
      "发现能力不需要权限，调用才需要，这和 MCP 的 `list_tools` / `call_tool` 分开是同一个道理。",
      "Card 里不要放内部地址和敏感信息。"),
            ("工具描述来自对方的 Agent Card",
      "远程团队改了能力或描述，编排方重新发现即可，不用改代码。",
      "选错 agent 的责任落在对方的 description 质量上，所以 Card 要有版本和维护人。"),
            ("`context_id` 在 `make_tool` 闭包里生成",
      "编排进程对每个远程 agent 保持一个会话，追问时对方能接上。",
      "一个进程只服务一个用户，所以够用。服务多用户时，要按「主 agent 的 thread_id × 远程 agent」维护映射；服务端也应把 user 拼进 thread_id，防止拿到别人的 context_id 后读到别人的对话。"),
            ("远程失败转成文字，不抛异常",
      "一个远程 agent 挂了或者超时，不会让整轮对话崩掉，主 agent 还能用另一个 agent 的结果回答，并说明哪一部分失败了。",
      "错误被吞成文字，要靠日志或链路追踪发现；生产环境还要加重试和熔断。"),
            ("`build_app` 工厂 + `APPS` 字典",
      "一个服务 = 一组配置，和「每个团队独立部署一份」的真实情况对应；新增 agent 只要加一项。",
      "两个服务跑在同一进程只是演示；真实部署时各自独立进程和独立的 task store。"),
        ],
        "steps": [
            ("01 · 服务端适配层：execute 的固定动作", "s16_a2a_agents.py",
      "顺序是固定的：确认有消息 → 取已有 task 或新建（新建要先入队）→ 用 `TaskUpdater` 推进状态 → 检查身份，不通过就 `reject` 并返回 → `start_work` → 用 `context_id` 作为 `thread_id` 调用 agent → `add_artifact` → `complete`。这里没有任何 LangChain 以外的业务逻辑。",
      ["LangChainExecutor"]),
            ("02 · 服务端横切：鉴权中间件", "s16_a2a_agents.py",
      "只拦 POST，也就是 JSON-RPC 调用；GET Agent Card 放行。令牌检查放在 HTTP 层，没有令牌的请求根本不会创建 task。",
      ["BearerAuth"]),
            ("03 · 服务端工厂：一组配置就是一个服务", "s16_a2a_agents.py",
      "`AgentCard` 是对外的全部承诺：名字、描述、版本、维护人、接口地址、能力和技能。`DefaultRequestHandler` 把 executor、task store 和 card 绑在一起；路由由 SDK 生成，最后套上鉴权中间件。",
      ["build_app", ("lines", "APPS = {", "}")]),
            ("04 · 编排方传输：send 只关心 task 状态", "s16_a2a.py",
      "请求里有两类信息：`message`（内容和 context_id）以及 `metadata.user`（替谁调用）。返回值按 task 状态分支：只有 COMPLETED 才取 artifacts，其他状态把原因带回去，交给主 agent 决定怎么回复。",
      ["send"]),
            ("05 · 编排方适配：Agent Card → 工具", "s16_a2a.py",
      "闭包里有三样东西：`card`（名字和描述）、`client`（带令牌的连接）、`context_id`（和这个远程 agent 的会话）。`ask_remote` 自己负责超时和错误转换，所以主 agent 看到的永远是一个正常返回文字的工具。",
      ["make_tool"]),
            ("06 · 编排方组装：发现 → 工具 → 主 agent", "s16_a2a.py",
      "两个 httpx 客户端：一个带令牌用于正常调用，一个不带令牌用于自检。连不上注册表里的地址就直接退出，不带着缺失的能力继续运行。最后的 `create_agent` 和 s15 的 supervisor 几乎一样，只是工具换成了远程 agent。",
      [("lines", "async def main", ")")]),
        ],
    },
}

