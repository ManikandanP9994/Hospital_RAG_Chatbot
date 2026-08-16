"""
app.py
FastAPI backend that exposes the hospital RAG chatbot over HTTP.

Run with:
    uvicorn app:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from rag_chain import get_chatbot, PERSIST_DIR

app = FastAPI(title="Hospital RAG Chatbot API")

# Allow the frontend (served from a different port/origin, e.g. during local
# dev on port 5500) to call this API. In production the frontend is served
# from this same app, so this mainly matters for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def build_index_if_missing():
    """On first boot (e.g. a fresh Hugging Face Space container), build the
    Chroma vector store automatically if it doesn't exist yet, so you don't
    have to SSH in and run ingest.py manually after every deploy."""
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print("No vector store found — running ingestion on startup...")
        import ingest
        ingest.main()


class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatTurn]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        bot = get_chatbot()
        history = [turn.model_dump() for turn in (req.history or [])]
        result = bot.ask(req.message, history=history)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend (index.html, etc.) from the same app/port.
# Mounted last, at "/", so it doesn't shadow the /health and /chat routes above.
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
