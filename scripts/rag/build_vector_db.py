from pathlib import Path
import shutil

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = PROJECT_ROOT / "data/rag_corpus"
VECTOR_DB_DIR = PROJECT_ROOT / "models/rag/chroma_bank_corpus"
REPORT_PATH = PROJECT_ROOT / "reports/rag_vector_db_summary.txt"

COLLECTION_NAME = "banking_policies"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chunk_text(text, chunk_size=500, overlap=80):
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def load_documents():
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Corpus folder not found: {CORPUS_DIR}")

    text_files = sorted(CORPUS_DIR.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(f"No .txt files found in: {CORPUS_DIR}")

    documents = []

    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8").strip()
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"{file_path.stem}_{index}",
                    "text": chunk,
                    "source": file_path.name,
                    "chunk_index": index,
                }
            )

    return documents


def main():
    documents = load_documents()

    print("Loaded documents/chunks:", len(documents))

    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model:")
    print(EMBEDDING_MODEL_NAME)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [doc["text"] for doc in documents]

    print("Creating embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Synthetic Arabic and English banking policy corpus"},
    )

    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        embeddings=embeddings,
        metadatas=[
            {
                "source": doc["source"],
                "chunk_index": doc["chunk_index"],
            }
            for doc in documents
        ],
    )

    print("Vector database created successfully.")
    print("Saved to:", VECTOR_DB_DIR)
    print("Collection count:", collection.count())

    test_questions = [
        "What documents are needed to open a savings account?",
        "متى تصل بطاقة الخصم البديلة؟",
        "هل الموافقة على القرض الشخصي مضمونة؟",
    ]

    report_lines = []
    report_lines.append("RAG Vector Database Summary")
    report_lines.append("=" * 60)
    report_lines.append(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    report_lines.append(f"Collection name: {COLLECTION_NAME}")
    report_lines.append(f"Number of stored chunks: {collection.count()}")
    report_lines.append("")

    for question in test_questions:
        query_embedding = embedding_model.encode([question]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
        )

        print("\nQuestion:", question)

        report_lines.append(f"Question: {question}")

        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            source = metadata["source"]

            print("Source:", source)
            print("Distance:", round(distance, 4))
            print("Text:", doc_text[:250].replace("\n", " "))

            report_lines.append(f"- Source: {source}")
            report_lines.append(f"  Distance: {round(distance, 4)}")
            report_lines.append(f"  Text: {doc_text[:250].replace(chr(10), ' ')}")

        report_lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nSummary report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
