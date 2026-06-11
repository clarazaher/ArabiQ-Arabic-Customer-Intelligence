# RAG Banking Assistant Model Card

## 1. System Overview

This component is part of **ArabiQ — Arabic Customer Intelligence Platform**.

The RAG assistant answers banking policy questions in English and Arabic by retrieving relevant text from a small synthetic banking policy corpus.

## 2. Purpose

The goal is to demonstrate how a customer-support assistant can answer questions from trusted banking documents instead of relying only on model memory.

Example questions:

- What documents are needed to open a savings account?
- How long does it take to get a replacement debit card?
- متى تصل بطاقة الخصم البديلة؟
- هل الموافقة على القرض الشخصي مضمونة؟

## 3. Data Source

The assistant uses a synthetic banking policy corpus stored in `data/rag_corpus/`.

The corpus includes Arabic and English documents about account opening, debit card replacement, and personal loans.

## 4. Retrieval Pipeline

The system works as follows:

1. Load Arabic and English banking policy documents.
2. Split the documents into chunks.
3. Convert chunks into vector embeddings.
4. Store the embeddings in ChromaDB.
5. Convert the user question into an embedding.
6. Retrieve the closest matching policy chunks.
7. Return an answer grounded in the retrieved source.

## 5. Embedding Model

The system uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

This model supports multilingual similarity, so it works for both English and Arabic retrieval.

## 6. Vector Database

The local vector database is stored in `models/rag/chroma_bank_corpus/`.

This folder is ignored by Git because it is generated locally and can be recreated.

## 7. Main Scripts

| Script | Purpose |
|---|---|
| `scripts/rag/create_synthetic_bank_corpus.py` | Creates the synthetic banking corpus |
| `scripts/rag/build_vector_db.py` | Builds the Chroma vector database |
| `scripts/rag/query_bank_rag.py` | Tests RAG retrieval from the terminal |
| `app/rag_app.py` | Runs the Streamlit RAG assistant |

## 8. Example Retrieval Results

| Question | Top Retrieved Source |
|---|---|
| What documents are needed to open a savings account? | `account_opening_en.txt` |
| How long does it take to get a replacement debit card? | `card_replacement_en.txt` |
| متى تصل بطاقة الخصم البديلة؟ | `card_replacement_ar.txt` |
| هل الموافقة على القرض الشخصي مضمونة؟ | `loan_policy_ar.txt` |

## 9. Streamlit App

Run the app with:

`streamlit run app/rag_app.py`

The app lets users ask Arabic or English banking questions and view the retrieved supporting sources.

## 10. Limitations

This is a prototype. The corpus is small and synthetic. The system retrieves policy text but does not yet include a full LLM answer generation layer. It may return long chunks instead of short answers, and it does not yet support user-uploaded documents.

## 11. Ethical and Safety Considerations

This assistant should support human decision-making, not replace it. For real banking use cases, the document corpus must be approved, regularly updated, and protected from private customer data exposure.

## 12. Reproducibility

Run these commands in order:

1. `python scripts/rag/create_synthetic_bank_corpus.py`
2. `python scripts/rag/build_vector_db.py`
3. `python scripts/rag/query_bank_rag.py`
4. `streamlit run app/rag_app.py`

To stop the Streamlit app, press `Control + C`.

## 13. Status

Phase 3 includes a bilingual synthetic banking corpus, multilingual embeddings, ChromaDB vector storage, terminal RAG testing, a Streamlit RAG UI, and this documentation.
