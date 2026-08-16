# Hospital RAG Chatbot (LangChain + ChromaDB + FastAPI)

A retrieval-augmented generation (RAG) chatbot for a hospital's FAQ/info
desk. It answers questions about departments, OPD timings, appointment
booking, billing, and facilities — grounded in your hospital's own
documents — and explicitly avoids giving medical advice.

## How it works

```
data/*.txt, *.pdf          (your hospital documents)
      │
      ▼
backend/ingest.py          chunks + embeds docs → stores in ChromaDB (local, persisted)
      │
      ▼
backend/rag_chain.py       retriever (Chroma) + prompt + ChatOpenAI → RAG chain
      │
      ▼
backend/app.py             FastAPI /chat endpoint wraps the chain
      │
      ▼
frontend/index.html        simple chat UI that calls the /chat endpoint
```

## Project structure

```
hospital-rag-chatbot/
├── backend/
│   ├── app.py            # FastAPI server (POST /chat)
│   ├── ingest.py          # builds the Chroma vector store from /data
│   ├── rag_chain.py       # RAG chain: retriever + prompt + LLM
│   ├── requirements.txt
│   └── .env.example
├── data/
│   └── hospital_info.txt  # sample hospital knowledge base (replace/add your own)
└── frontend/
    └── index.html         # chat UI (plain HTML/JS, no build step)
```

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed locally (this project runs fully
  local/free by default — no API key or billing required). If you'd
  rather use OpenAI's hosted models instead, see "Using OpenAI instead"
  below.

Pull the two models Ollama needs:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## 2. Backend setup

```bash
cd hospital-rag-chatbot/backend

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

No `.env` or API key is needed for the default (Ollama) setup.

## 3. Build the vector database

Add or replace files in `data/` with your hospital's real FAQs, policies,
department lists, etc. (`.txt` and `.pdf` are supported out of the box).
Then run the ingestion script once — this embeds the documents and
persists them to a local ChromaDB folder (`backend/chroma_db/`):

```bash
python ingest.py
```

Re-run this any time your source documents change.

## 4. Run the backend

```bash
uvicorn app:app --reload --port 8000
```

Check it's alive: open `http://localhost:8000/health` → should return `{"status":"ok"}`.

Test the chat endpoint directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the cardiology OPD timings?"}'
```

## 5. Run the frontend

The frontend is a single static HTML file — no build step needed. Two options:

**Option A — just open it:**
Double-click `frontend/index.html` (or open it in a browser directly).

**Option B — serve it (recommended, avoids some browser CORS quirks):**

```bash
cd hospital-rag-chatbot/frontend
python -m http.server 5500
```

Then visit `http://localhost:5500` in your browser.

Make sure the backend (step 4) is running at the same time — the frontend
calls `http://localhost:8000/chat` (edit the `API_URL` constant at the top
of the `<script>` in `index.html` if your backend runs elsewhere).

## Customizing for your hospital

- **Add real content**: drop more `.txt`/`.pdf` files into `data/` (e.g.
  doctor bios, insurance lists, department-specific FAQs), then re-run
  `python ingest.py`.
- **Adjust the system prompt**: edit `SYSTEM_PROMPT` in `backend/rag_chain.py`
  to change tone, add more safety rules, or support another language.
- **Retrieval tuning**: change `search_kwargs={"k": 4}` in `rag_chain.py`
  to retrieve more/fewer chunks per question.
- **Swap the LLM/embeddings model**: both are set in `rag_chain.py`
  (`ChatOpenAI(model=...)`, `OpenAIEmbeddings(model=...)`).

## Using OpenAI instead of Ollama (optional, paid)

This project defaults to fully local models via Ollama (free, no API key).
If you'd rather use OpenAI's hosted models instead (e.g. for higher
quality answers), you'll need an OpenAI account with billing/credits set
up at platform.openai.com/settings/organization/billing.

1. Install the OpenAI integration package:
   ```bash
   pip install langchain-openai
   ```
2. In `ingest.py` and `rag_chain.py`, replace:
   ```python
   from langchain_community.embeddings import OllamaEmbeddings
   embeddings = OllamaEmbeddings(model="nomic-embed-text")
   ```
   with:
   ```python
   from langchain_openai import OpenAIEmbeddings
   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
   ```
3. In `rag_chain.py`, replace:
   ```python
   from langchain_community.chat_models import ChatOllama
   self.llm = ChatOllama(model="llama3.2", temperature=0.2)
   ```
   with:
   ```python
   from langchain_openai import ChatOpenAI
   self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
   ```
4. Copy `.env.example` to `.env` and set your `OPENAI_API_KEY`.
5. Re-run `python ingest.py` (embeddings must match between ingest and
   query, so switching models requires rebuilding the vector store).

## Important safety note

This chatbot is designed to answer **informational** questions only
(timings, booking, billing, facilities). The system prompt explicitly
instructs the model to refuse medical diagnoses or treatment advice and
redirect patients to a doctor or the emergency line instead. Before
deploying this in a real hospital setting, have it reviewed by your
clinical/compliance team, and consider adding conversation logging,
authentication, and rate limiting for production use.
