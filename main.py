from fastapi import FastAPI
from handler import startup, generate

app = FastAPI()

@app.on_event("startup")
def load():
    startup()

@app.post("/chat")
def chat(payload: dict):
    message = payload.get("message", "")
    if not message:
        return {"error": "No message provided"}
    return {"response": generate(message)}

