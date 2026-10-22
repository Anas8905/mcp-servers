import asyncio
from fastmcp import Client, FastMCP


mcp_server = FastMCP("MCP Server 2")

# Tools

@mcp_server.tool()
def tool_add(a: int, b: int) -> int:
    """Add two numbers together and return the sum."""
    return a + b

@mcp_server.tool()
def tool_to_upper(text: str) -> str:
    """"Return the incoming text in all capital letters."""
    return text.upper()


# MCP Client / Server block

async def mcp_client(method: str, payload: dict):
    async with Client(mcp_server) as client:

        for tool in await client.list_tools():
            print(tool.name, ' | ', tool.description)
            print("   schema: ", tool.input_schema, "\n")

        response = await client.call_tool(method, payload)

    return response

response = asyncio.run(mcp_client(method="tool_add", payload={"a":18, "b":24}))

print(response)
