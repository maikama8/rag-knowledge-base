# RAG Knowledge Base

A simple Retrieval-Augmented Generation (RAG) application that lets you upload documents and ask questions with citations. Designed as a clean public portfolio project showing modern AI engineering practices.

## Features (Planned / Scaffold)

- Upload PDFs or text documents
- Chunking + embeddings
- Hybrid or vector search
- Question answering with source citations
- Basic evaluation notes in README
- Simple web UI

## Tech Stack (Suggested)

- Python 3.11+
- FastAPI or Streamlit
- LangChain / LlamaIndex or plain embeddings
- FAISS / Chroma / pgvector for vector store
- OpenAI or local embeddings

## Getting Started

```bash
git clone https://github.com/maikama8/rag-knowledge-base.git
cd rag-knowledge-base
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
rag-knowledge-base/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py
│   └── rag.py
└── data/
```

## Author

**Abdulrasheed Oseni** ([@maikama8](https://github.com/maikama8))

## License

MIT
