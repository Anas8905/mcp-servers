import secrets, threading, asyncio, uvicorn, requests, time, os, hashlib, jwt
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastmcp import FastMCP

from pydantic import BaseModel
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

# Generate user login

def generate_new_secret():
    Path(".env").write_text(f"SECRET_KEY={secrets.token_hex(32)}\nALGORITHM=HS256\n")
    print("[INFO] :: New secret token generated!")

if not Path(".env").exists():
    generate_new_secret()

load_dotenv()

SECRET_KEY, ALGORITHM = os.getenv("SECRET_KEY"), os.getenv("ALGORITHM")
API_BASE = "http://127.0.0.1:8000"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify(plain, hashed):
    return hash_password(plain) == hashed

DB = {"user1": {"username": "user1", "hashed_password": hash_password("password123")}}


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
oauth2 = OAuth2PasswordBearer(tokenUrl="login")   # "tokens arrive in the Authorization header"

class ChatSchema(BaseModel):
    message: str


# ---------- 1. CREATE LOGIN ----------
def create_access_token(username: str) -> str:
    return jwt.encode({"sub": username,
                       "exp": datetime.now(timezone.utc) + timedelta(minutes=30)},
                      SECRET_KEY, algorithm=ALGORITHM)


# ---------- 2. LOGIN ----------
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = DB.get(form.username)

    if not user or not verify(form.password, user["hashed_password"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")

    return {"access_token": create_access_token(form.username), "token_type": "bearer"}


# ---------- 3. GUARD ----------
def get_current_user(token: str = Depends(oauth2)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])   # signature + expiry
    except jwt.PyJWTError:                                                # any failure -> 401
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Could not validate credentials")
    return payload["sub"]


# ---------- 4. ENDPOINTS ----------
class ChatIn(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatSchema, user: str = Depends(get_current_user)) -> dict:
    #  FastAPI runs the guard first. If it raises 401, this function never executes.
    result = await agent.run(request.message)
    return {"user": user, "reply": result.output}


    # ---------- 5. START SERVER ----------
def start_server(app, port):
    """Start uvicorn on a background thread and DON'T return until it is really listening."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    t = threading.Thread(target=run, daemon=True)
    t.start()

    for _ in range(100):
        if server.started:
            return f"http://127.0.0.1:{port}"
        if not t.is_alive():                   # thread crashed -> say so LOUDLY.
            raise RuntimeError(               # otherwise our requests would silently hit
                "uvicorn thread died - see the traceback above")
        time.sleep(0.3)

    raise RuntimeError(f"server did not start on port {port} within 30s")


if __name__ == "__main__":

    BASE = start_server(app, 8001)

    assert requests.post(f"{BASE}/login", data={}).status_code != 404,     "/login is missing - BASE points at the wrong server. Restart the runtime."

    # 1. no token -> 401
    print("no token :", requests.post(f"{BASE}/chat", json={"message": "hi"}).status_code)

    # 2. wrong password -> 401     (data= not json= : OAuth2 expects a FORM body)
    print("bad pwd  :", requests.post(f"{BASE}/login",
                                      data={"username": "user1", "password": "WRONG"}).status_code)

    # 3. correct login -> a JWT    (credentials must match DB above)
    r = requests.post(f"{BASE}/login", data={"username": "user1", "password": "password123"})
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"   # show the body, not a KeyError
    TOKEN = r.json()["access_token"]
    print("login    :", r.status_code)

    # 4. use it: one header, on every request
    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.post(f"{BASE}/chat", headers=headers,
                      json={"message": "What is 17 * 23 + 100?"}, timeout=120)
    print("chat     :", r.json())

    # 5. forged token -> 401 (they don't have our SECRET_KEY, so the signature won't match)
    forged = jwt.encode({"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(minutes=30)},
                        "f" * 64, algorithm=ALGORITHM)
    print("forged   :", requests.post(f"{BASE}/chat", headers={"Authorization": f"Bearer {forged}"},
                                      json={"message": "hi"}).status_code)
