import os
import time
import logging
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from scalar_fastapi import get_scalar_api_reference

# --- 1. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Security & Config (Fail-Fast) ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not GEMINI_API_KEY or not PINECONE_KEY or not PINECONE_ENV:
    raise RuntimeError("Missing critical environment variables. Application cannot start.")

os.environ["PINECONE_API_KEY"] = PINECONE_KEY
os.environ["PINECONE_ENV"] = PINECONE_ENV

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)
INDEX_NAME = "docsoft"
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY)

# FIX: Global shared client, reused across all requests
vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB limit

# --- 3. Seeded API Key "Database" ---
# In a real enterprise app, this would be a DB lookup (e.g., Postgres/Redis).
# For this portfolio demo, we use a hardcoded mapping to prove the server-side validation mechanism.
API_KEY_DB = {
    "sk_docsoft_user_001": "tenant_abc",
    "sk_docsoft_user_002": "tenant_xyz"
}

# --- 4. Pydantic Models ---
class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class UploadResponse(BaseModel):
    message: str
    chunks: int
    filename: str

class SourceChunk(BaseModel):
    page: int | str
    content: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]

class ErrorResponse(BaseModel):
    detail: str

# --- 5. Auth & Session Dependency ---
def get_tenant_id(authorization: str = Header(...)):
    """Validates the API key and resolves the server-side tenant ID."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header. Use 'Bearer <api_key>'.")
    
    token = authorization.split(" ")[1]
    
    # FIX: Actual server-side validation. The client cannot guess or pass arbitrary namespaces.
    tenant_id = API_KEY_DB.get(token)
    
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized API key.")
    
    return tenant_id

# --- 6. FastAPI App Initialization ---
app = FastAPI(
    title="DocSoft Enterprise API",
    summary="A production-grade, multi-tenant RAG microservice for document Q&A.",
    description="""
    ## DocSoft Enterprise AI Q&A API
    
    Multi-tenant RAG microservice using Google Gemini and Pinecone.
    All requests require a `Authorization: Bearer <api_key>` header. 
    The server resolves the API key to a strict Pinecone namespace to guarantee data isolation.
    
    ### Demo API Keys
    * `sk_docsoft_user_001` (Tenant A)
    * `sk_docsoft_user_002` (Tenant B)
    """,
    version="1.3.0",
    contact={
        "name": "Utkarsh Pandey",
        "url": "https://utkarshizm.github.io/Portfolio/",
    },
    openapi_tags=[
        {"name": "Health", "description": "Service health and status endpoints."},
        {"name": "Documents", "description": "Endpoints for uploading and managing enterprise documents."},
        {"name": "Q&A", "description": "Endpoints for querying the knowledge base."},
        {"name": "Session", "description": "Endpoints for managing session data."},
        {"name": "Documentation", "description": "API documentation endpoints."}
    ],
    docs_url=None, 
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/docs", tags=["Documentation"], include_in_schema=False)
def get_scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="DocSoft API Documentation"
    )

# --- 7. API Endpoints ---
@app.get("/", tags=["Health"])
def home():
    return {"status": "DocSoft API is up and running!"}

@app.post(
    "/upload", 
    tags=["Documents"],
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Upload and index a PDF document"
)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id)
):
    start_time = time.perf_counter() 
    logger.info(f"Received file upload: {file.filename} for tenant: {tenant_id}")
    tmp_path = None
    
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is supported.")

        contents = await file.read(MAX_FILE_SIZE + 1)
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        def process_and_store():
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            vectorstore.add_documents(chunks, namespace=tenant_id)
            return len(chunks)

        chunk_count = await run_in_threadpool(process_and_store)
        
        os.unlink(tmp_path)
        
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Indexed {chunk_count} chunks for tenant {tenant_id} in {elapsed_time:.4f}s")
        
        return UploadResponse(
            message="Document indexed successfully in Pinecone", 
            chunks=chunk_count,
            filename=file.filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path) 
        raise HTTPException(status_code=500, detail="Internal server error processing document.")

@app.post(
    "/ask", 
    tags=["Q&A"],
    response_model=AnswerResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ask a question about the uploaded documents"
)
async def ask_question(
    q: Question,
    tenant_id: str = Depends(get_tenant_id)
):
    start_time = time.perf_counter() 
    logger.info(f"Processing question for tenant {tenant_id}: {q.question}")
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "namespace": tenant_id})
        sources = await retriever.ainvoke(q.question)
        
        context_text = "\n\n---\n\n".join([d.page_content for d in sources])
        
        prompt = PromptTemplate(
            template="Answer based ONLY on this context:\n{context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"]
        )
        
        chain = prompt | llm
        raw_response = await chain.ainvoke({"context": context_text, "question": q.question})
        
        if isinstance(raw_response.content, list):
            answer_text = "".join([part.get("text", "") for part in raw_response.content if isinstance(part, dict) and part.get("type") == "text"])
        else:
            answer_text = raw_response.content

        if not answer_text.strip():
            logger.warning(f"Sanitization resulted in empty answer for question: {q.question}")

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Query processed in {elapsed_time:.4f} seconds")
        
        return AnswerResponse(
            answer=answer_text,
            sources=[SourceChunk(page=d.metadata.get("page", "?"), content=d.page_content[:200]) for d in sources]
        )
    except Exception as e:
        logger.error(f"Error answering question '{q.question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error generating answer.")

@app.delete("/session", tags=["Session"], summary="Delete all documents for the current tenant")
async def delete_session_data(tenant_id: str = Depends(get_tenant_id)):
    """Clears all vectors associated with the provided tenant ID (namespace)."""
    try:
        await run_in_threadpool(
            vectorstore.delete, 
            delete_all=True, 
            namespace=tenant_id
        )
        logger.info(f"Deleted all documents for tenant: {tenant_id}")
        return {"message": "Session data deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting data for tenant {tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error deleting data.")
