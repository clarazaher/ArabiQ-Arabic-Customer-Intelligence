from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_DB_DIR = PROJECT_ROOT / "models/rag/chroma_bank_corpus"

COLLECTION_NAME = "banking_policies"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve_context(question, collection, embedding_model, n_results=2):
    query_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    retrieved_items = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved_items.append(
            {
                "text": text,
                "source": metadata["source"],
                "distance": round(distance, 4),
            }
        )

    return retrieved_items


def create_answer(retrieved_items):
    best_item = retrieved_items[0]

    answer = (
        "Based on the retrieved banking policy:\n\n"
        f"{best_item['text']}\n\n"
        f"Source: {best_item['source']}"
    )

    return answer


def main():
    st.set_page_config(
        page_title="ArabiQ Banking RAG Assistant",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 ArabiQ Banking RAG Assistant")
    st.write(
        "Ask a banking policy question in English or Arabic. "
        "The assistant retrieves the most relevant policy text from the banking corpus."
    )

    if not VECTOR_DB_DIR.exists():
        st.error(
            "Vector database not found. Please run: "
            "python scripts/rag/build_vector_db.py"
        )
        return

    with st.spinner("Loading RAG system..."):
        embedding_model = load_embedding_model()
        collection = load_collection()

    question = st.text_area(
        "Your question",
        value="What documents are needed to open a savings account?",
        height=100,
    )

    n_results = st.slider(
        "Number of retrieved sources",
        min_value=1,
        max_value=3,
        value=2,
    )

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving relevant policy text..."):
            retrieved_items = retrieve_context(
                question,
                collection,
                embedding_model,
                n_results=n_results,
            )

        st.subheader("Answer")
        st.write(create_answer(retrieved_items))

        st.subheader("Retrieved Sources")

        for item in retrieved_items:
            with st.expander(f"{item['source']} | distance: {item['distance']}"):
                st.write(item["text"])

    st.divider()

    st.caption(
        "This is a retrieval-based prototype. It answers only from the synthetic banking policy corpus."
    )


if __name__ == "__main__":
    main()
