DocSoft: Enterprise AI Q&A API

A production-grade Retrieval-Augmented Generation (RAG) microservice built with FastAPI. It allows users to upload PDF documents and ask natural language questions, returning answers grounded strictly in the source text with page-level citations to eliminate LLM hallucinations.

🌐 Live API: docsoft-tueu.onrender.com📖 Interactive API Docs (Swagger UI): docsoft-tueu.onrender.com/docs
📊 Production Performance Metrics

This API is deployed on Render and monitored via internal high-resolution time.perf_counter() logging. Latency is kept strictly in server logs to prevent internal timing surface area exposure.
Metric	Latency
p50	1.44s
p95	1.50s

Latency includes PDF text extraction, semantic similarity search across the Pinecone vector database, and Google Gemini LLM generation.
🏗️ System Architecture

graph TD    Client([👤 User / Client App])    API[FastAPI Backend]    Loader[PyPDFLoader & Text Splitter]    Embed[Google Gemini Embeddings]    Pinecone[(Pinecone Cloud Vector DB)]    LLM[Google Gemini LLM]    Client -->|1. Upload PDF| API    API -->|2. Extract & Chunk| Loader    Loader -->|3. Vectorize| Embed    Embed -->|4. Store Vectors| Pinecone        Client -->|5. Ask Question| API    API -->|6. Similarity Search| Pinecone    Pinecone -->|7. Return Top-K Chunks| API    API -->|8. Context + Question| LLM    LLM -->|9. Generated Answer| API    API -->|10. JSON Response + Citations| Client

🛠️ Tech Stack

     Backend: FastAPI, Uvicorn (ASGI)
     AI/LLM: LangChain, Google Gemini API (gemini-3.5-flash-lite & gemini-embedding-001)
     Vector Database: Pinecone (Managed Cloud)
     Infrastructure: Docker, GitHub Actions (CI/CD), Render (Cloud Hosting)

🚀 Key Features

    Decoupled Architecture: Separated the RAG backend from the UI, allowing any frontend (Streamlit, React, mobile app) to connect via standard REST API.
    Cloud Vector Search: Migrated from local ChromaDB to Pinecone for scalable, persistent, and fast similarity search.
    Response Sanitization: Defensively strips Google's internal safety/tracking metadata (signature blocks) from the LLM response before returning it to the client.
    Observability: perf_counter latency tracking and empty-answer warning logs to catch silent LLM failures.

📡 API Endpoints

     GET /
         Health check to verify the API is running.
     POST /upload
         Input: multipart/form-data (PDF file)
         Action: Extracts text, chunks it, embeds it, and stores it in Pinecone.
     POST /ask
         Input: {"question": "Your question here?"}
         Output: {"answer": "LLM generated text", "sources": [{"page": 0, "content": "..."}]}

📦 Run Locally
Prerequisites

     Python 3.10+
     A Google Gemini API Key
     A Pinecone Account (to create an index with 3072 dimensions)

Installation

    Clone the repository:
    bash
     
      
     
     
    git clone https://github.com/utkarshizm/DocSoft.git
    cd DocSoft
     
     

    Install dependencies:
    bash
     
      
     
     
    pip install -r requirements.txt
     
     

    Set Environment Variables:
    bash
     
      
     
     
    export GEMINI_API_KEY="your_google_api_key"
    export PINECONE_API_KEY="your_pinecone_key"
    export PINECONE_ENV="gcp-starter"
     
     

    Run the FastAPI server:
    bash
     
      
     
     
    uvicorn main:app --reload --port 8000
     
     

    Open your browser and navigate to:
    http://localhost:8000/docs

Run with Docker

    Build the image:
    bash
     
      
     
     
    docker build -t docsoft-api .
     
     
    Run the container:
    bash
     
      
     
     
    docker run -p 8000:8000 --env-file .env docsoft-api
     
     

⚙️ CI/CD Pipeline

This repository includes a GitHub Actions workflow (.github/workflows/deploy.yml) that automatically:

    Triggers on every push/PR to the main branch.
    Sets up a Python environment.
    Installs dependencies to verify requirements.txt resolves correctly.
    Prepares the code for Docker build (which Render handles automatically on deploy).

© 2026 Utkarsh Pandey. Designed & Developed with passion.
```
    
     
