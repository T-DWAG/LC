# 环境

在项目根目录执行。环境里安装 Python 3.12，后面的包都进这个解释器。

```powershell
conda create -n LC python=3.12 -y
conda activate LC
python -m pip install -U langchain-openai langchain-core langgraph python-dotenv
python -c "import langchain_openai, langchain_core, langgraph, dotenv; print('ok')"
```

后面的课还要用到这些包（s10 SQLite 检查点、s12/s13 中间件、s13 deepagents 版、s14 MCP、s16 A2A）：

```powershell
python -m pip install -U "langchain[mcp]>=1.4" langgraph-checkpoint-sqlite "a2a-sdk[http-server]>=1.3" uvicorn deepagents
```

`conda activate` 报错时，先执行一次 `conda init powershell`，重开终端再激活。

# 环境变量

三个值给后续代码用。模型是 DeepSeek V4.1 Flash，接口里的 ID 是 `deepseek-flash`。

```powershell
$enc = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllText("$PWD\.env", @"
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.aicodewith.ai/v1
OPENAI_MODEL=deepseek-flash
"@, $enc)
```

# .gitignore

```powershell
[IO.File]::WriteAllText("$PWD\.gitignore", @"
.venv/
.env
__pycache__/
*.pyc
.DS_Store
"@, $enc)
```
