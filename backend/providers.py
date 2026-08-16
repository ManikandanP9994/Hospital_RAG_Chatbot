"""
providers.py
Central place that decides which embedding model and LLM to use, based on
the LLM_PROVIDER environment variable.

LLM_PROVIDER=ollama  (default) -> free, local, requires Ollama running locally
LLM_PROVIDER=openai            -> hosted, requires OPENAI_API_KEY, used for deployment

This lets the exact same codebase run locally with Ollama and in
production (e.g. Hugging Face Spaces) with OpenAI, just by changing one
environment variable / secret.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


def get_embeddings():
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")


def get_llm():
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    else:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3.1", temperature=0.2)
