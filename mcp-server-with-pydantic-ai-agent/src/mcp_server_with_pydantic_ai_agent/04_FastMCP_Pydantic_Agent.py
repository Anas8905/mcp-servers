import asyncio

from fastmcp import FastMCP
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


# Model

def ollama_model(name):
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
    )

model = ollama_model("qwen3.8:latest")


# Tools

mcp_server = FastMCP("FileServer")

@mcp_server.tool()
def write_file(filename: str, content: str) -> str:
    """Save content into a text file. Use this tool whenever asked to save, write or create a file with a given content. Create a camel-case filename based on the content if the user does not specify one."""
    print(f"[INFO] :: [TOOL RUN] :: wrtie_file({filename!r}, {len(content)} characters")
    Path(filename).write_text(content)
    return f"Saved {filename} with {len(content)} content length."


# MCP Agent

agent = Agent(
    model,
    system_prompt="You are a helpful assistant that can use your tool to save content to text files.",
    toolsets = [MCPToolset(mcp_server)]
)

response = asyncio.run(agent.run("Write a Haiku about the computer game series The Last of Us."))
print(response)
