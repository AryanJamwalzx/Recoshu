# ReCoShU — Find Your Fit 👟

An AI-powered shoe recommendation and customer support assistant built using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **Smart Product Recommendations:** Search Nike and Adidas shoe catalogs using natural-language queries.
- **Customer Support:** Retrieve relevant information from FAQs, return policies, and shoe-care documents.
- **Semantic Search:** Generate embeddings using Sentence Transformers and retrieve relevant documents from Pinecone.
- **Reranking:** Improve retrieval relevance using a cross-encoder reranker.
- **Query Understanding:** Use an LLM to interpret queries and identify user constraints.
- **REST API:** Expose the RAG pipeline through FastAPI.
- **Observability:** Expose application metrics through a Prometheus-compatible endpoint.
- **Docker Support:** Run the application in a containerized environment.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| LLM | Mistral |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | Pinecone |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Data Processing | Pandas |
| Monitoring | Prometheus |
| Containerization | Docker |

## 🏗️ Architecture

1. **Data ingestion:** Load product catalogs, FAQs, and policy documents.
2. **Preprocessing:** Convert source data into documents and chunks.
3. **Embedding generation:** Create vector representations using Sentence Transformers.
4. **Vector storage:** Store vectors and metadata in Pinecone.
5. **Query analysis:** Interpret user intent and extract relevant constraints.
6. **Retrieval and reranking:** Retrieve candidate documents and rank them by relevance.
7. **Response generation:** Generate responses grounded in retrieved context.
8. **API and monitoring:** Serve queries through FastAPI and expose application metrics.

## 📁 Project Structure

```text
RAG_ASST/
├── app/
│   ├── api/
│   ├── Pipeline/
│   ├── Retrieval/
│   ├── Embeddings/
│   ├── Vector_Store/
│   ├── Preprocessing/
│   ├── loaders/
│   ├── ingestion/
│   ├── context_builder/
│   ├── Evaluation/
│   ├── Observability/
│   └── data/
├── UI/
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## ⚙️ Getting Started

### Prerequisites

- Python 3.12 or a compatible version
- Pinecone API credentials
- Mistral API credentials
- Docker (optional)

### 1. Clone the repository

```bash
git clone https://github.com/AryanJamwalzx/Recoshu.git
cd Recoshu
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and configure the credentials using the exact variable names expected by the application.

**Never commit your `.env` file or expose API keys.**

### 5. Run the application

```bash
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## 🔌 API Endpoints

| Endpoint | Description |
|---|---|
| `/` | Web interface |
| `/health` | Application health check |
| `/query` | Submit a query to the RAG pipeline |
| `/metrics/` | Prometheus-compatible application metrics |

## 🐳 Run with Docker

Build the Docker image:

```bash
docker build -t recoshu-rag:latest .
```

Run the container:

```bash
docker run --rm --env-file .env -p 8000:8000 recoshu-rag:latest
```

Then visit [http://localhost:8000](http://localhost:8000).

## 🧪 Testing

Run the automated test suite:

```bash
python -m pytest -v -m "not live_llm"
```

The recorded development test run completed with **31 passed, 1 deselected, and 0 failed**. Results may vary with code or environment changes.

## 📊 Evaluation

The project includes a curated query evaluation suite and retrieval constraint checks. Evaluation results should be interpreted in the context of the evaluation dataset and current implementation.

## 🔐 Security

- Store API keys in environment variables.
- Keep `.env` out of version control.
- Review data licensing and redistribution permissions before making the repository public.
- Avoid exposing credentials in logs or documentation.

## 🔮 Future Improvements

- Expand retrieval and recommendation evaluation.
- Add automated retrieval regression tests.
- Improve latency and caching.
- Extend production monitoring and tracing.

---

**ReCoShU** — Exploring RAG, semantic search, reranking, API development, and observability.
