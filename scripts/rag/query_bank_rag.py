from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = PROJECT_ROOT / "models/rag/chroma_bank_corpus"
REPORT_PATH = PROJECT_ROOT / "reports/rag_query_examples.txt"

COLLECTION_NAME = "banking_policies"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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


def create_grounded_answer(question, retrieved_items):
    context = retrieved_items[0]["text"]
    source = retrieved_items[0]["source"]

    answer = (
        "Based on the retrieved banking policy, the answer is:\n\n"
        f"{context}\n\n"
        f"Source: {source}"
    )

    return answer


def main():
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector database not found at {VECTOR_DB_DIR}. "
            "Run scripts/rag/build_vector_db.py first."
        )

    print("Loading embedding model:")
    print(EMBEDDING_MODEL_NAME)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    questions = [
        "What documents are needed to open a savings account?",
        "How long does it take to get a replacement debit card?",
        "متى تصل بطاقة الخصم البديلة؟",
        "هل الموافقة على القرض الشخصي مضمونة؟",
    ]

    report_lines = []
    report_lines.append("RAG Query Examples")
    report_lines.append("=" * 60)

    for question in questions:
        print("\n" + "=" * 80)
        print("Question:", question)

        retrieved_items = retrieve_context(question, collection, embedding_model)
        answer = create_grounded_answer(question, retrieved_items)

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved sources:")
        for item in retrieved_items:
            print(f"- {item['source']} | distance: {item['distance']}")

        report_lines.append("")
        report_lines.append(f"Question: {question}")
        report_lines.append("")
        report_lines.append("Answer:")
        report_lines.append(answer)
        report_lines.append("")
        report_lines.append("Retrieved sources:")

        for item in retrieved_items:
            report_lines.append(f"- {item['source']} | distance: {item['distance']}")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nQuery examples saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
