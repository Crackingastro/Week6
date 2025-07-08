# rag_task3.py

import os
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from typing import List, Tuple
from dotenv import load_dotenv

# ——————————————————————————————————————————————————————————
# CONSTANTS
# ——————————————————————————————————————————————————————————
CHROMA_DIR      = "../vector_store/chroma_db"
COLLECTION_NAME = "cfpb_complaints"
MODEL_ID        = "deepseek-ai/DeepSeek-V3-0324"
PROMPT_TEMPLATE = """
You are a senior data analyst at CrediTrust with deep expertise in consumer complaint analysis.
Your role: transform raw complaint excerpts into clear, data-driven insights.

For each question:
1. Analysis: List key themes or metrics identified in the context.
3. Insight: Summarize actionable conclusions or root causes.
4. Recommendation: Suggest next investigative steps if applicable.

Use only the provided context. If there's insufficient information, respond:
"I’m sorry, I don’t have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
""".strip()

# ——————————————————————————————————————————————————————————
# INIT PERSISTENT CHROMA CLIENT & EMBEDDINGS & LLM CLIENT
# ——————————————————————————————————————————————————————————
_client = PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

load_dotenv(dotenv_path=".env")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
hhf_token = HF_TOKEN

inference_client = InferenceClient(
    model=MODEL_ID,
    token=hhf_token
)

# ——————————————————————————————————————————————————————————
# CORE RAG FUNCTIONS
# ——————————————————————————————————————————————————————————
def embed_text(text: str) -> List[float]:
    """Convert text to embedding vector."""
    return embeddings.embed_query(text)


def retrieve(question: str, k: int = 5) -> List[str]:
    """Retrieve top-k similar document chunks from Chroma."""
    q_emb = embed_text(question)
    results = _collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents"]
    )
    return results["documents"][0]


def generate_answer(question: str,
                    contexts: List[str],
                    max_tokens: int = 512) -> str:
    """Generate an answer using DeepSeek LLM."""
    ctx = "\n\n".join(contexts)
    prompt = PROMPT_TEMPLATE.format(context=ctx, question=question)
    response = inference_client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE.split('Context:')[0].strip()},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def run_rag(question: str, k: int = 5) -> Tuple[str, List[str]]:
    """End-to-end RAG: retrieve contexts and generate answer."""
    contexts = retrieve(question, k)
    answer = generate_answer(question, contexts)
    return answer, contexts

# ——————————————————————————————————————————————————————————
# DEMO with company-specific questions
# ——————————————————————————————————————————————————————————
if __name__ == "__main__":
    questions = [
        "Are people unhappy with BMO? If so why?",
        "What issues do users report with our mobile app?",
        "Which feature is most requested by users?",
        "Which companies have the highest rate of 'Closed without relief'?"
    ]

    for question in questions:
        answer, contexts = run_rag(question)
        print("Question:", question)
        print("Answer:", answer)
        print("Retrieved Contexts:")
        for idx, ctx in enumerate(contexts, 1):
            print(f"{idx}. {ctx}")
        print("=" * 150)
