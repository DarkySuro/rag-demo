# Open-Domain Wikipedia RAG System

A production-style Retrieval-Augmented Generation (RAG) system that answers open-domain questions by retrieving relevant Wikipedia passages and generating grounded answers with citations using an LLM served by Groq.

Built as a learning-focused capstone project — the pipeline is developed step-by-step in Jupyter notebooks before being consolidated into a FastAPI backend and Streamlit UI.

---

## Features

- Semantic search over a Wikipedia article subset using dense vector embeddings
- Local, persistent vector storage with ChromaDB (no external vector DB service required)
- Fast, free LLM inference via the Groq API
- REST API for querying the RAG pipeline
- Interactive Streamlit UI with source citations
- Retrieval and answer-quality evaluation harness

---

## Tech Stack

| Component | Tool |
|---|---|
| Environment / package manager | [uv](https://github.com/astral-sh/uv) |
| Data source | Hugging Face `datasets` (Wikipedia snapshot) |
| Chunking | Custom / LangChain text splitter |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database | [ChromaDB](https://www.trychroma.com/) |
| LLM generation | [Groq API](https://console.groq.com) (Llama 3.x models) |
| Backend API | FastAPI + Uvicorn |
| UI | Streamlit |
| Evaluation | Custom metrics / RAGAS |

---

## Project Structure

```
rag-capstone/
├── notebooks/
│   ├── 01_data_prep.ipynb        # Load and explore Wikipedia subset
│   ├── 02_chunking.ipynb         # Split articles into chunks
│   ├── 03_embeddings_index.ipynb # Embed chunks and store in ChromaDB
│   ├── 04_retrieval.ipynb        # Query ChromaDB, inspect retrieval quality
│   ├── 05_generation.ipynb       # RAG prompt construction + Groq LLM calls
│   └── 06_evaluation.ipynb       # Retrieval and answer-quality evaluation
├── data/
│   ├── raw/                      # Raw downloaded Wikipedia data
│   └── processed/                # Cleaned/chunked data
├── src/                          # Reusable pipeline modules (extracted from notebooks)
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── generator.py
├── api/
│   └── main.py                   # FastAPI application
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── eval/
│   ├── eval_set.json             # Labeled Q&A pairs for evaluation
│   └── evaluate.py
├── chroma_db/                    # Persisted Chroma vector store (gitignored)
├── .env                          # API keys (gitignored)
├── pyproject.toml
└── README.md
```

---

## Setup

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and initialize the project

```bash
git clone <your-repo-url>
cd rag-capstone
uv sync
```

### 3. Add your Groq API key

Create a free API key at [console.groq.com](https://console.groq.com), then create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

### 4. Launch Jupyter

```bash
uv run jupyter notebook
```

Run the notebooks in order (`01` → `06`) to build the pipeline from data ingestion through evaluation.

### 5. Run the API (after the pipeline is validated in notebooks)

```bash
uv run uvicorn api.main:app --reload
```

### 6. Run the UI

```bash
uv run streamlit run app/streamlit_app.py
```

---

## Pipeline Overview

```
Wikipedia subset (HF datasets)
        │
        ▼
   Chunking (≈300–500 tokens, with overlap)
        │
        ▼
   Embedding (sentence-transformers)
        │
        ▼
   ChromaDB (persistent vector store)
        │
        ▼
 User query → embed → similarity search → top-k chunks
        │
        ▼
   Groq LLM (context + query → grounded answer)
        │
        ▼
   FastAPI → Streamlit UI (answer + source citations)
```

---

## Evaluation

The `eval/` module measures:

- **Retrieval quality**: recall@k — does the retriever surface the passage containing the ground-truth answer?
- **Answer faithfulness**: does the generated answer stay grounded in retrieved context, or hallucinate?
- **Latency**: end-to-end response time per query

---

## Roadmap / Future Improvements

- [ ] Hybrid search (BM25 + dense vectors)
- [ ] Cross-encoder reranking of retrieved chunks
- [ ] Query result caching
- [ ] Docker Compose setup for one-command local deployment
- [ ] Scale from a Wikipedia subset to the full corpus

---

## License

MIT
