import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# ────────────────────────────────────────────────────
# Constants & Configuration
# ────────────────────────────────────────────────────
CHROMA_DIR = "../vector_store/chroma_db"
COLLECTION_NAME = "cfpb_complaints"
MODEL_ID = "deepseek-ai/DeepSeek-V3-0324"

PROMPT_TEMPLATE = """
You are a senior data analyst at CrediTrust with deep expertise in consumer complaint analysis.
Your role: transform raw complaint excerpts into clear, data-driven insights.

For each question:
1. Analysis: List key themes or metrics identified in the context.
2. Insight: Summarize actionable conclusions or root causes.
3. Recommendation: Suggest next investigative steps if applicable.

Use only the provided context. If there's insufficient information, respond:
"I'm sorry, I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
""".strip()

# Load environment variables
load_dotenv(dotenv_path=".env")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ────────────────────────────────────────────────────
# Initialize Components
# ────────────────────────────────────────────────────
@st.cache_resource
def init_chroma_client():
    return PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )

@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def init_llm_client():
    return InferenceClient(
        model=MODEL_ID,
        token=HF_TOKEN
    )

chroma_client = init_chroma_client()
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
embeddings = init_embeddings()
inference_client = init_llm_client()

# ────────────────────────────────────────────────────
# Core RAG Functions
# ────────────────────────────────────────────────────
def embed_text(text: str) -> List[float]:
    """Convert text to embedding vector."""
    return embeddings.embed_query(text)

@st.cache_data(show_spinner="Searching knowledge base...")
def retrieve(question: str, k: int = 5) -> List[str]:
    """Retrieve top-k similar document chunks from Chroma."""
    q_emb = embed_text(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results["documents"][0], results["metadatas"][0]

def generate_answer(question: str, contexts: List[str]) -> str:
    """Generate an answer using DeepSeek LLM."""
    ctx = "\n\n".join([f"Document {i+1}:\n{text}" for i, text in enumerate(contexts)])
    prompt = PROMPT_TEMPLATE.format(context=ctx, question=question)
    
    response = inference_client.chat_completion(
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE.split('Context:')[0].strip()},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.2
    )
    
    # Handle streaming response if needed
    if hasattr(response, 'choices'):
        return response.choices[0].message.content.strip()
    return ""

def run_rag(question: str, k: int = 5) -> Tuple[str, List[Tuple[str, dict]]]:
    """End-to-end RAG: retrieve contexts and generate answer."""
    documents, metadatas = retrieve(question, k)
    answer = generate_answer(question, documents)
    return answer, list(zip(documents, metadatas))

# ────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────
st.set_page_config(page_title="📊 CrediTrust RAG Chat", layout="wide")
st.title("📊 CrediTrust Complaint Analysis")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources", expanded=False):
                for i, (doc, meta) in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    **Document {i}**  
                    **Complaint ID:** {meta.get('complaint_id', 'N/A')}  
                    **Product:** {meta.get('product', 'N/A')}  
                    **Issue:** {meta.get('issue', 'N/A')}  
                    **State:** {meta.get('state', 'N/A')}  
                    **Company:** {meta.get('company', 'N/A')}
                    """)
                    st.caption(doc[:300] + "..." if len(doc) > 300 else doc)
                    st.divider()

# Sidebar controls
with st.sidebar:
    st.header("🔧 Settings")
    k = st.slider("Number of documents to retrieve", 1, 10, 5)
    
    st.header("💬 Example Questions")
    example_questions = [
        "Are people unhappy with BMO? If so why?",
        "What issues do users report with our mobile app?",
        "Which feature is most requested by users?",
        "Which companies have the highest rate of 'Closed without relief'?"
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.user_input = q

# User input
if prompt := st.chat_input("Ask about consumer complaints...") or getattr(st.session_state, 'user_input', None):
    if 'user_input' in st.session_state:
        prompt = st.session_state.user_input
        del st.session_state.user_input
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get RAG response
    with st.spinner("Analyzing complaints..."):
        answer, sources = run_rag(prompt, k)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("📚 Sources", expanded=False):
            for i, (doc, meta) in enumerate(sources, 1):
                st.markdown(f"""
                **Document {i}**  
                **Complaint ID:** {meta.get('complaint_id', 'N/A')}  
                **Product:** {meta.get('product', 'N/A')}  
                **Issue:** {meta.get('issue', 'N/A')}  
                **State:** {meta.get('state', 'N/A')}  
                **Company:** {meta.get('company', 'N/A')}
                """)
                st.caption(doc[:300] + "..." if len(doc) > 300 else doc)
                st.divider()