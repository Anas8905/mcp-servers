from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "add",
    "params": {"a": 2, "b": 3}
}

ok = {"jsonrpc": "2.0", "id": 1, "result": 5}
unknown_function = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}}


def ollama_model(name):
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
    )

model = ollama_model("qwen3.8:latest")


def add(a: int, b: int) -> int:
    return a + b

def to_upper(text: str) -> str:
    return text.upper()

FUNCTIONS = {"add": add, "to_upper": to_upper}

def handle(request: dict):
    function =  FUNCTIONS.get(request["method"])

    if function is None:
        return unknown_function

    result = function(**request["params"])
    ok["result"] = result

    return ok


# DEBUG
print(handle({"jsonrpc": "2.0", "id": 1, "method": "add", "params": {"a": 21, "b": 21}}))
print(handle({"jsonrpc": "2.0", "id": 1, "method": "to_upper", "params": {"text": "hi there!"}}))
print(handle({"jsonrpc": "2.0", "id": 1, "method": "subtract", "params": {"a": 21, "b": 21}}))
