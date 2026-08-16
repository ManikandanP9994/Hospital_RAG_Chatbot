"""
ingest.py
Loads all documents from the /data folder, splits them into chunks,
embeds them, and persists them into a local ChromaDB vector store.

Run this once (or whenever hospital documents change):
    python ingest.py
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from providers import get_embeddings

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "hospital_knowledge_base"


def load_documents():
    docs = []

    # Load all .txt files
    txt_loader = DirectoryLoader(
        DATA_DIR, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs.extend(txt_loader.load())

    # Load all .pdf files (if any are added later, e.g. policy documents)
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs.extend(pdf_loader.load())

    return docs


def main():
    print("Loading documents from:", DATA_DIR)
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunk(s).")

    embeddings = get_embeddings()

    # Wipe any old persisted DB so re-running ingest.py doesn't duplicate data
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"Vector store persisted to: {PERSIST_DIR}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
