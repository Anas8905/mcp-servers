import threading, asyncio, uvicorn, requests, time

from fastapi import FastAPI
from fastmcp import FastMCP

from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

API_BASE = "http://127.0.0.1:8000"

# Model

def ollama_model(name):
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
    )

model = ollama_model("qwen3.8:latest")


# Tools

mcp_server = FastMCP("Utils")


@mcp_server.tool()
def calculate(expression: str) -> str:
    """Evaluate a simple math expression, e.g. '22 * 2 * ((1+1) / 2)."""
    print(f"[INFO] [TOOL RUN] :: calculate({expression!r})")
    allowed_char = "0123456789+-*/., ()"
    if not set(expression) <= set(allowed_char): #whitelist input
        return  f"[Error] :: Only {allowed_char} allowed!"
    return str(eval(expression))


# MCP Agent

agent = Agent(model,
    system_prompt="You are a helpful assistant that can evaluate simple mathematical equations using it's tools.",
    toolsets=[MCPToolset(mcp_server)])

# REST API

app = FastAPI()

class ChatSchema(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatSchema):
    result = await agent.run(request.message)
    return {"reply": result.output}

server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

threading.Thread(target=run_server, daemon=True).start()

while not server.started:
    time.sleep(0.3)

health_check = requests.get(f"{API_BASE}/health").json()

print(f"[INFO] [API] :: {health_check}")



response = requests.post(
    f"{API_BASE}/chat",
    json={"message": "What is (16 + 68) / 2 ?"},
    timeout=120
)

print(response.json())
