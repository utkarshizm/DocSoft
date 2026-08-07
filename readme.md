# DocSoft — Production RAG API for Document Q&A

DocSoft is a Retrieval-Augmented Generation (RAG) microservice that lets users upload documents (PDF, text) and ask natural-language questions grounded strictly in their content — reducing LLM hallucination by design and returning cited source passages with every answer.

**Live API docs:** https://docsoft-tueu.onrender.com/docs

---

## The problem

General-purpose LLM chat can confidently generate answers that aren't actually supported by a source document. DocSoft addresses this by constraining generation to retrieved context only:

1. Uploaded documents are chunked and embedded into a vector database
2. A user's question triggers retrieval of only the most relevant chunks
3. The LLM is instructed to answer strictly from that retrieved context
4. The response includes the exact source passages used, with page numbers

## Architecture

```
Document upload ──▶ Chunk & embed (LangChain) ──▶ Pinecone vector DB
                                                          │
                                                          ▼ retrieval
User query ──▶ FastAPI backend (Gemini LLM call) ──▶ Answer + cited sources

Deployment (on every push to main):
  Docker build ──▶ GitHub Actions CI/CD ──▶ Render (hosts the FastAPI backend)
```

- **Ingestion:** documents are split into chunks, embedded, and indexed in Pinecone for persistent vector search
- **Query:** a question hits the FastAPI `/ask` endpoint, retrieves top-matching chunks from Pinecone, and passes them as grounding context to the LLM
- **Response:** LLM output is sanitized (internal metadata stripped) before returning a clean answer with page-level source attribution
- **Multi-tenancy:** server-side API key resolution maps each caller to an isolated Pinecone namespace, preventing cross-tenant data access
- **Deployment:** every push to `main` runs through GitHub Actions, builds a Docker image, and deploys to Render with no manual steps

## Features

- Conversational Q&A grounded in uploaded documents, with source attribution (page numbers) on every response
- Multi-tenant data isolation via namespace-scoped API keys
- Cloud vector search via Pinecone (persistent, scales beyond local FAISS/Chroma)
- Dockerized FastAPI backend with automatic OpenAPI docs (`/docs`)
- CI/CD pipeline: push to `main` → automated build → deploy to Render
- Latency instrumentation (`time.perf_counter()`) around the retrieval + generation path

## Tech stack

| Layer | Tools |
|---|---|
| LLM & orchestration | Google Gemini API, LangChain |
| Vector database | Pinecone |
| Backend | FastAPI, Python |
| Deployment | Docker, GitHub Actions, Render |
| Frontend (optional) | Streamlit |

## Performance testing & observability

**Concurrent load test.** An async load test (`httpx`, 5 concurrent workers) was run against the live API to characterize behavior under simultaneous traffic. This surfaced a real bottleneck: Google Gemini's free-tier rate limit (15 RPM) was being exceeded, and because the client library didn't explicitly handle `429` responses, they surfaced to callers as unhandled `500` errors.

**Baseline latency benchmark.** To characterize normal-case performance independent of that rate limit, a sequential benchmark (1 request at a time, n=12) was run against the live endpoint:

| Metric | Latency |
|---|---|
| p50 | ~1.48s |
| p95 | ~2.16s |

Latency covers the full retrieval + generation path: Pinecone semantic search plus Gemini LLM generation, timed server-side with `time.perf_counter()`.

*Note: this is a small-sample baseline (n=12), not a high-confidence SLA — sufficient to characterize typical response time, not to bound tail latency precisely.*

## API usage

**Ask a question:**

```bash
curl -X POST 'https://docsoft-tueu.onrender.com/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <your_api_key>' \
  -d '{
    "question": "What is my name and what university did I attend?"
  }'
```

**Response:**

```json
{
  "answer": "Based on the provided context...",
  "sources": [
    { "page": 0, "content": "..." }
  ]
}
```

Full interactive docs (Swagger UI) are available at [`/docs`](https://docsoft-tueu.onrender.com/docs).

## Running locally

```bash
# clone the repo
git clone https://github.com/utkarshizm/docsoft.git
cd docsoft

# install dependencies
pip install -r requirements.txt

# set environment variables (see below)
cp .env.example .env

# run the app
uvicorn main:app --reload
```

Or with Docker:

```bash
docker build -t docsoft .
docker run -p 8000:8000 --env-file .env docsoft
```

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index to use |

## Project structure

```
docsoft/
├── main.py              # FastAPI app, /ask and /upload endpoints
├── ingestion.py         # chunking + embedding pipeline
├── requirements.txt
├── Dockerfile
├── .github/workflows/   # CI/CD pipeline
└── README.md
```

## Roadmap

- [ ] Rate-limit handling / backoff for downstream LLM 429s
- [ ] Multi-document session support
- [ ] Streaming responses
- [ ] Automated latency dashboard (p50/p95 over time, larger sample size)

## Author

**Utkarsh Pandey**
· pandeyutkarsh060@gmail.com

## License

MIT
