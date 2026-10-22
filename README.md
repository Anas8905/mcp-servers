# Model Context Protocol (FastMCP & LangChain)

> 🎓 **Personal training repo** — hands-on notes on building & consuming MCP
> servers. Pre-work for the [instar-mcp-server](https://github.com/mpolinowski/instar-mqtt-mcp).

* [MCP Server Foundation](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-09-08--mcp-server-foundation/2026-09-08)
* [MCP Protocol Deepdive](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-09-09--mcp-protocol-deepdive/2026-09-09)
* [MCP Server Deployment](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-10-08--mcp-server-deployment/2026-10-08)
* [MCP Gateway](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-10-09--mcp-gateway/2026-10-09)
* [INSTAR MQTT Camera AI Agent Interface](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-09-10--instar-mqtt-mcp-server/2026-09-10)

---

## 🛠️ What is MCP used for?

**MCP (Model Context Protocol)** is an open, JSON-RPC-based standard for how
LLM-powered apps talk to the outside world. Instead of wiring a bespoke
integration per tool or data source, an **MCP server** exposes a set of
*tools* (callable actions), *resources* (readable data), and *prompts* — and
any **MCP client** (Claude, Cursor, a LangChain agent, …) can use them.
One standard plug: **model in, capability out.**

| Layer | What it is |
| :-- | :-- |
| **MCP server** 🖥️ | Exposes tools / resources / prompts over *stdio* or *streamable-HTTP*. |
| **MCP client** 🖱️ | Connects to a server, lists its tools, calls them. |
| **FastMCP** ⚡ | Python framework atop the official `mcp` SDK — build servers in a few lines. |
| **LangChain ↔ MCP** 🤖 | The adapter that turns MCP tools into LangChain tools (client side). |

This repo is that stack in miniature: **build servers (FastMCP)** + **use them
(LangChain / raw SDK)**, then **package & deploy** them.

---

## 📚 Research — a tour of the folders

**The learning ladder (suggested reading order):**

```
🔬 deepdive-mcp          learn the protocol + both kinds of client
🌦️ weather-node-mcp      the same thing in TypeScript (not just Python)
📁 file-access-python-mcp build a real, clean FastMCP server
📦 mcp-server-deployment → package it as an installable library
🚪 mcp-server-gateway    → aggregate many servers behind one endpoint
🐳 mcp-docker-deployment → ship it in a container
🤖 mcp-server-with-pydantic-ai-agent → consume it from the other side:
                                        Pydantic AI agents, from raw JSON-RPC to a secured REST API
```

| 📁 Repo | What's inside | 🎯 Takeaway |
| :-- | :-- | :-- |
| **deepdive-mcp** 🔬 | Three parallel scenarios — ① raw `mcp` SDK over `stdio`, ② FastMCP over stdio **+** `streamable-HTTP`, ③ a LangChain MCP-adapter client hitting a community `duckduckgo` server. | The 3 layers of MCP in one place: **SDK server → FastMCP → LangChain client.** The core “what / where / how” playground. |
| **weather-node-mcp** 🌦️ | Minimal MCP server in **TypeScript** using the official `@modelcontextprotocol` v2 client **+** server and `zod`. One tool (`get-alerts`, US NWS) + one resource, and a demo client. | MCP in the **other language** — and that server *and* client can live in the same folder. |
| **file-access-python-mcp** 📁 | Production-flavoured FastMCP server: exposes a local folder of text docs as **9 tools** (search / list / glob / get / edit / create / delete) over stdio **and** streamable-HTTP+CORS, with `--allow-*` permission flags. | The **cleanest example here** — a solid template (domain / tools / transport split + e2e tests). Copy this one. |
| **mcp-server-deployment** 📦 | `agent_terminal_tools`: a FastMCP server of shell & file tools (run command, list, glob, grep, read/write/append/delete/copy/move) packaged as an **installable Python package** (`uv` build + entry point); client demos included, test suite still to come. | How to turn a server into a **package** others can `pip install` / `uvx`. |
| **mcp-server-gateway** 🚪 | A FastMCP **gateway**: one server that `mount(create_proxy(...))`s several upstreams (DuckDuckGo **+** `agent_terminal_tools`) and re-exposes them over streamable-HTTP. | The **aggregate / proxy pattern** — many servers behind a single endpoint. |
| **mcp-docker-deployment** 🐳 | The gateway above, **containerised**: `python:3.14-slim` + `uv`, pre-installs a `uvx` community server, runs it over streamable-HTTP. | How to **ship** an MCP server as a Docker image. |
| **mcp-server-with-pydantic-ai-agent** 🤖 | An 8-step training ladder that builds to a finished app: `01` a hand-rolled **JSON-RPC** dispatcher (the wire format under MCP) → `02` the **MCP protocol** on it (`initialize` / `tools/list` / `tools/call`, JSON-Schema `inputSchema`) → `03` the same server in **FastMCP** + its client → then agents driving it: `04` a single tool that **writes a haiku to a file** (source of the loose `*-haiku.txt` in the folder root) → `05` a **multi-tool loop** (weather → web search → hotel selection) → `06` an **orchestrator/worker** pairing where researcher & actor agents are wrapped as tools of a manager → `07` the agent behind **FastAPI** (`/health` + `/chat`, uvicorn in a thread) → `08` the **JWT & password-hash** building blocks. `main.py` is the product: OAuth2 login, a JWT guard, the MCP `calculate` tool, and a self-test that exercises every auth failure. `reference.py` = the original Colab course the ladder was distilled from. | The **client-side companion** to the server tour — **how an agent consumes MCP tools**, then productises into a secured REST API. |

> 🚧 `weaviate-mcp/` is a vendored **Go** MCP server for the Weaviate vector DB —
> **WIP, not yet published**, so it's left out of the tour.


[![M8ven Score](https://m8ven.ai/badge/mcp/mpolinowski-mcp-servers-179eo7?v=0ae6390de2478363cb0c766e22ebcb38)](https://m8ven.ai/mcp/mpolinowski-mcp-servers-179eo7?s=readme)

---

## 📜 License & audit

- **Licensed [MIT](LICENSE)** — one repo-root license covers every sub-project.
- **[SECURITY_AUDIT.md](SECURITY_AUDIT.md)** records a security audit's findings
  and this repo's verdict on each — including the `grep` "handler mutates" claim,
  verified against the source and found to be a **false positive**, and the
  "0/33 tools in tests" claim, which overstates the real (partial) gap.
