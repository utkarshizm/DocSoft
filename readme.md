# DocSoft — Production RAG API for Document Q&A

DocSoft is a Retrieval-Augmented Generation (RAG) microservice that lets you upload documents (PDF, text) and ask natural-language questions grounded strictly in their content — eliminating LLM hallucinations by design.

**Live API docs:** https://docsoft-tueu.onrender.com/docs

---

## Overview

Traditional LLM chat can confidently invent answers that aren't in your source documents. DocSoft solves this by:

1. Chunking and embedding uploaded documents into a vector database
2. Retrieving only the most relevant chunks for a given question
3. Forcing the LLM to answer strictly from that retrieved context
4. Returning the answer alongside the exact source passages used

## Architecture

```
Document upload ──▶ Chunk & embed (LangChain) ──▶ Pinecone vector DB (cloud index)
                                                          │
                                                          ▼ retrieval
User query ──▶ FastAPI backend (Gemini LLM call) ──▶ Answer + cited sources
                       │
                       ▼
              Docker + GitHub Actions CI/CD ──▶ Render (production deploy)
```

- **Ingestion:** documents are split into chunks and embedded, then indexed in Pinecone for persistent, sub-2s vector search
- **Query:** a question hits the FastAPI `/ask` endpoint, retrieves the top-matching chunks from Pinecone, and passes them as context to the LLM
- **Response:** the LLM's raw output is sanitized (internal metadata/signatures stripped) before returning a clean answer with page-level source attribution
- **Deployment:** every push runs through GitHub Actions CI/CD, builds a Docker image, and deploys to Render with zero manual steps

## Features

- Conversational Q&A grounded in uploaded documents — no hallucinated answers
- Source attribution with page numbers on every response
- Cloud vector search via Pinecone (persistent, scales beyond local FAISS/Chroma)
- Dockerized FastAPI backend with automatic OpenAPI docs (`/docs`)
- CI/CD pipeline: push to `main` → automated build → deploy to Render
- Performance instrumentation (`time.perf_counter()`) for latency monitoring

## Tech stack

| Layer | Tools |
|---|---|
| LLM & orchestration | Google Gemini API, LangChain |
| Vector database | Pinecone |
| Backend | FastAPI, Python |
| Deployment | Docker, GitHub Actions, Render |
| Frontend (optional) | Streamlit |

## Performance

Measured over 30 production queries via Render logs:

| Metric | Value |
|---|---|
| p50 latency | 1.44s |
| p95 latency | 1.50s |

Latency is timed with `time.perf_counter()` around the retrieval + generation path and logged server-side.

## API usage

**Ask a question:**

```bash
curl -X POST 'https://docsoft-tueu.onrender.com/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
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

- [ ] API key / rate limiting on public endpoints
- [ ] Multi-document session support
- [ ] Streaming responses
- [ ] Automated latency dashboard (p50/p95 over time)

## Author

**Utkarsh Pandey**
· pandeyutkarsh060@gmail.com

## License

MIT
