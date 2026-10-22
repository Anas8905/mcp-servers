import base64, json, secrets, os, jwt, hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

def generate_new_secret():
    Path(".env").write_text(
        f"SECRET_KEY={secrets.token_hex(32)}\n"
        "ALGORITHM=HS256\n"
    )
    print("[INFO] :: New secret token generated!")

if not Path(".env").exists():
    generate_new_secret()

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
WRONG_KEY = "f" * 64

print("[DEBUG] :: API secret\n")
print(Path(".env").read_text())

payload = {
    "sub": "user",
    "exp": datetime.now(timezone.utc) + timedelta(minutes=60)
}

token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

print(f"[DEBUG] :: User token created {token}")

def peek(chunk):
    return json.loads(base64.urlsafe_b64decode(chunk + "=" * (-len(chunk) % 4)))

header, payload, signature = token.split(".")
print(f"[DEBUG] :: Header {peek(header)}")
print(f"[DEBUG] :: Payload {peek(payload)}")

print(f"[DEBUG] :: Valid signature {jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])}")

try:
    jwt.decode(token, WRONG_KEY, algorithms=[ALGORITHM])
except jwt.InvalidSignatureError as e:
    print(f"[DEBUG] :: Test false signature error: {e}")

expired_token = jwt.encode(
    {
        "sub": "bad user",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1)
    },
    SECRET_KEY,
    algorithm=ALGORITHM
)

try:
    jwt.decode(expired_token, WRONG_KEY, algorithms=[ALGORITHM])
except jwt.InvalidSignatureError as e:
    print(f"[DEBUG] :: Test expired token error: {e}")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify(plain, hashed):
    return hash_password(plain) == hashed

DB = {"user": {"hashed_password": hash_password("password123")}}

print(f"[DEBUG] :: Verifying user: {DB['user']}")
print(f"[DEBUG] :: Verifying password: {verify('password123', DB['user']['hashed_password'])}")
print(f"[DEBUG] :: Verifying wrong password: {verify('321drowssap', DB['user']['hashed_password'])}")
