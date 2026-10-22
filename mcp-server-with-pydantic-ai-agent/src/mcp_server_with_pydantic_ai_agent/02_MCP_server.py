import json

# Tools

request_template = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "add",
    "params": {"a": 2, "b": 3}
}

ok = {"jsonrpc": "2.0", "id": 1, "result": 0}
unknown_tool = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}}
unknown_tool_call = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Tool call not defined"}}

def tool_add(a: int, b: int) -> int:
    return a + b

def tool_to_upper(text: str) -> str:
    return text.upper()

Toolkit = {"tool_add": tool_add, "tool_to_upper": tool_to_upper}

# MCP Server

Tools = [
     {
         "name": "add",
         "description": "Add two numbers together and return the sum.",
         "inputSchema": {
             "type": "object",                        # arguments always arrive as a dict
             "properties": {"a": {"type": "number"},  # argument a -> a number
                            "b": {"type": "number"}}, # argument b -> a number
             "required": ["a", "b"],                  # both mandatory
         },
     },
     {
         "name": "shout",
         "description": "Return the given text in capital letters with an exclamation mark.",
         "inputSchema": {
             "type": "object",
             "properties": {"text": {"type": "string"}},
             "required": ["text"],
         },
     },
]


def mcp_server(request: dict) -> dict:
    method, params, id = request["method"], request.get("params", {}), request["id"]

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "MCP Server 1", "version": "1.0"}
        }
    elif method == "tools/list":
        result = {
            "tools": Tools
        }
    elif method == "tools/call":
        try:
            out = Toolkit[params["name"]](**params["arguments"])
            result = {
                "content": [{"type": "text", "text": str(out)}]
            }
        except:
            result = unknown_tool

    else:
        result = unknown_tool_call

    ok["id"] = id
    ok["result"] = result

    return ok


def mcp_call(method: str, params: dict | None = None, _id: list[int] = [0]):
    _id[0] += 1

    request = {
        "jsonrpc": "2.0",
        "id": _id[0],
        "method": method,
        "params": params or {}
    }

    response = mcp_server(request)

    print(f"--> {json.dumps(request)}")
    print(f"<-- {json.dumps(response)}\n")

# DEBUG
mcp_call("initialized")
mcp_call("tools/list")
mcp_call("tools/call", {"name": "tool_add", "arguments": {"a": 26, "b": 16}})
mcp_call("tools/call", {"name": "tool_subtract", "arguments": {"a": 26, "b": 16}})
mcp_call("tools/call", {"name": "tool_to_upper", "arguments": {"text": "hey there!"}})
mcp_call("tools/calls", {"name": "tool_to_upper", "arguments": {"text": "hey there!"}})
