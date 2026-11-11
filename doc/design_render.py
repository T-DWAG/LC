"""把 design_notes.DESIGNS 渲染成 HTML，插入 v2 页面的 s14–s16。由 sync-tutorial-code.py 调用。"""
from __future__ import annotations

import ast
import html
import re
import textwrap
from pathlib import Path
from typing import Callable

DESIGN_CSS = (
    "<style>/* design-css */"
    ".design{border-top:1px solid var(--line);padding-top:8px}"
    ".design-h{font-size:15px;margin:26px 0 6px;color:var(--ink)}"
    ".design-seq{margin:8px 0;padding-left:22px;color:#3d4842}"
    ".design-seq li{margin:6px 0}"
    ".design-seq code,.design td code{background:var(--soft);padding:1px 4px;border-radius:3px}"
    ".step-file{font:12px/1.4 ui-monospace,Consolas,monospace;color:var(--muted)!important;margin-top:10px!important}"
    "</style>"
)


def inline(text: str) -> str:
    """讲解文字：先转义，再把 `代码` 和 **加粗** 换成标签。"""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def excerpt(root: Path, name: str, spec) -> str:
    """按 design_notes 的写法从 code/ 源码截取片段，保证讲解里的代码和源码一致。"""
    source = (root / "code" / name).read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    if isinstance(spec, str) and spec.isidentifier():
        defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        tree = ast.parse(source)
        node = next((n for n in tree.body if isinstance(n, defs) and n.name == spec), None)
        if node is None:
            raise RuntimeError(f"{name} 里找不到顶层定义 {spec}")
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return "".join(lines[start - 1:node.end_lineno]).rstrip()
    if isinstance(spec, str):
        start_prefix = end_prefix = spec
    else:
        start_prefix, end_prefix = spec[1], spec[2]
    try:
        begin = next(i for i, line in enumerate(lines) if line.lstrip().startswith(start_prefix))
        finish = next(i for i in range(begin, len(lines)) if lines[i].lstrip().startswith(end_prefix))
    except StopIteration:
        raise RuntimeError(f"{name} 里找不到片段 {spec!r}") from None
    # 片段从缩进中间开始时，去掉共同缩进
    return textwrap.dedent("".join(lines[begin:finish + 1])).rstrip()


def rows(items) -> str:
    return "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in items)


def render(root: Path, note: dict, render_source: Callable[[str, list[str]], str]) -> str:
    out = [
        '<div class="coder-guide design"><h3>工程设计：代码是怎么组织的</h3>',
        f'<p>{inline(note["intro"])}</p>',
        '<h4 class="design-h">模块与职责</h4><table class="cmp">',
        "<tr><th>模块</th><th>包含</th><th>职责</th><th>边界</th></tr>",
        rows(note["files"]),
        "</table>",
    ]
    if "compare" in note:
        cmp = note["compare"]
        head = "".join(f"<th>{inline(h)}</th>" for h in cmp["head"])
        out.append(f'<h4 class="design-h">三种编排横向对比</h4><table class="cmp"><tr>{head}</tr>{rows(cmp["rows"])}</table>')
    out.append('<h4 class="design-h">一次调用的完整链路</h4><ol class="design-seq">')
    out.extend(f"<li>{inline(s)}</li>" for s in note["sequence"])
    out.append("</ol>")
    out.append('<h4 class="design-h">设计决策与代价</h4><table class="cmp">')
    out.append("<tr><th>决策</th><th>为什么这样做</th><th>代价 / 生产环境要补的</th></tr>")
    out.append(rows(note["decisions"]))
    out.append("</table>")
    out.append('<h4 class="design-h">按设计顺序对照代码</h4>')
    out.append("<p>片段直接截取自源码，按设计顺序排列，不是独立可运行的程序；完整文件见上方。</p>")
    for title, name, text, specs in note["steps"]:
        code = "\n\n".join(excerpt(root, name, spec) for spec in specs)
        out.append(
            f'<div class="guide-step"><div><h4>{html.escape(title)}</h4><p>{inline(text)}</p>'
            f'<p class="step-file">{html.escape(name)}</p></div>'
            f"<pre><code>{render_source(code, [])}</code></pre></div>"
        )
    out.append("</div>")
    return "".join(out)


def sync(page: str, root: Path, designs: dict, render_source) -> str:
    """把讲解块插到每章 notes 之后、运行说明之前；已有就替换，可重复执行。"""
    for chapter, note in designs.items():
        block = f"<!-- design:{chapter}:start -->{render(root, note, render_source)}<!-- design:{chapter}:end -->"
        marker = re.compile(rf"<!-- design:{chapter}:start -->.*?<!-- design:{chapter}:end -->", re.DOTALL)
        if marker.search(page):
            page = marker.sub(lambda _: block, page, count=1)
            continue
        section = re.search(rf'<section id="{chapter}">.*?</section>', page, re.DOTALL)
        if section is None:
            raise RuntimeError(f"找不到章节 {chapter}")
        run = section.group(0).rfind('<p class="run">')
        if run < 0:
            raise RuntimeError(f"{chapter} 里找不到运行说明")
        at = section.start() + run
        page = page[:at] + block + page[at:]
    if "/* design-css */" not in page:
        page = page.replace("</head>", DESIGN_CSS + "</head>", 1)
    return page
