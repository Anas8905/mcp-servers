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




# JSON-RPC Server

def json_rpc_server(request: dict):
    tool =  Toolkit.get(request["method"])

    if tool is None:
        return unknown_tool

    result = tool(**request["params"])
    ok["result"] = result

    return ok


# DEBUG
print(json_rpc_server({"jsonrpc": "2.0", "id": 1, "method": "tool_add", "params": {"a": 21, "b": 21}}))
print(json_rpc_server({"jsonrpc": "2.0", "id": 1, "method": "tool_to_upper", "params": {"text": "hi there!"}}))
print(json_rpc_server({"jsonrpc": "2.0", "id": 1, "method": "subtract", "params": {"a": 21, "b": 21}}))
