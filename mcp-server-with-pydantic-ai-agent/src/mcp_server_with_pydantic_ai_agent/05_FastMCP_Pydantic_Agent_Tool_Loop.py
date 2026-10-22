import urllib.parse, aiohttp, python_weather, asyncio
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from datetime import datetime


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

mcp_server = FastMCP("TravelServer")

@mcp_server.tool()
async def check_weather(location: str) -> str:
    """Get tge current weather for a location."""
    print(f"[INFO] :: [TOOL RUN] :: check_weather({location!r})")
    async with python_weather.Client(unit=python_weather.METRIC) as client:
        temperature = await client.get(location)
        return f"{location}: {temperature}"

@mcp_server.tool()
async def search_web(query: str) -> str:
    """Search the web. Use this tool to find travel destinations."""
    print(f"[INFO] :: [TOOL RUN] :: search_web({query!r})")
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"

    async with aiohttp.ClientSession() as search:
        async with search.get(url, headers={"User-Agent": "Mozilla/5.0"}) as request:
            soup = BeautifulSoup(await request.text(), "html.parser")

    hits = [a.text.strip() for a in soup.find_all("a", class_="result__snippet")][:3]
    return "\n".join(hits) or "No results."

@mcp_server.tool()
async def select_hotel(hotel_name: str, dates: str) -> str:
    """Select a hotel. Only call this AFTER you have found a real hotel name."""
    print(f"  [TOOL] book_hotel({hotel_name!r})")
    return f"CONFIRMED: {hotel_name} for {dates}."



# MCP Agent

agent = Agent(
    model,
    system_prompt=
    "You are a travel assistant. Always: "
    "1) check the weather, 2) search the web for REAL hotels, 3) select one you found. "
    "Never invent a hotel name. And dont ask any followup question",
    toolsets = [MCPToolset(mcp_server)]
)

today = datetime.now().strftime("%A, %B %d, %Y")

response = asyncio.run(agent.run(f"Today is {today}. Plan a day trip to Kuala Lumpur and seleect a hotel for me next Saturday."))
print(response)
