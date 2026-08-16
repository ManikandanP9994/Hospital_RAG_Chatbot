"""
rag_chain.py
Builds the retrieval-augmented generation chain used by the hospital chatbot:
ChromaDB retriever -> prompt (with hospital-safety rules) -> ChatOpenAI LLM.
"""

import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from providers import get_embeddings, get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "hospital_knowledge_base"

SYSTEM_PROMPT = """You are the virtual assistant for City Care General Hospital.

Use ONLY the context below to answer the patient's question about hospital
services, departments, timings, booking, billing, or facilities.

Rules:
- If the answer is not contained in the context, say you don't have that
  information and suggest calling the reception at +1-555-0100.
- NEVER provide a medical diagnosis, treatment advice, or interpretation of
  symptoms or test results. If asked, politely redirect the patient to
  consult a doctor directly or call the Emergency Line for urgent issues.
- Keep answers concise, friendly, and factual.
- If it's a medical emergency, tell the patient to call the Emergency Line
  immediately.

Context:
{context}
"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _format_history(history):
    """history: list of {"role": "user"|"assistant", "content": str}"""
    messages = []
    for turn in history or []:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


class HospitalRAGChatbot:
    def __init__(self):
        embeddings = get_embeddings()
        self.vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})

        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        self.chain = (
            {
                "context": lambda x: _format_docs(self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str, history=None) -> dict:
        chat_history = _format_history(history)
        answer = self.chain.invoke({"question": question, "chat_history": chat_history})
        sources = self.retriever.invoke(question)
        return {
            "answer": answer,
            "sources": [doc.page_content[:200] for doc in sources],
        }


# Singleton instance reused across requests
_chatbot_instance = None


def get_chatbot() -> HospitalRAGChatbot:
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = HospitalRAGChatbot()
    return _chatbot_instance
