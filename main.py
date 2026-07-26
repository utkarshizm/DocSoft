import os
import time
import logging
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- 1. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="DocSoft Enterprise API")

# --- 2. Config & Globals ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set!")

# Setup Pinecone Environment Variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
os.environ["PINECONE_ENV"] = os.getenv("PINECONE_ENV", "")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)
INDEX_NAME = "docsoft"

# --- 3. Pydantic Models ---
class Question(BaseModel):
    question: str

# --- 4. API Endpoints ---
@app.get("/")
def home():
    logger.info("Health check endpoint hit.")
    return {"status": "DocSoft API is up and running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}")
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load & Split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        # Embed & Store in Pinecone
        vectorstore = PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
        
        os.unlink(tmp_path) # cleanup
        logger.info(f"Successfully indexed {len(chunks)} chunks from {file.filename}")
        return {"message": "Document indexed successfully in Pinecone", "chunks": len(chunks)}
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(q: Question):
    # Use perf_counter for high-resolution, monotonic timing
    start_time = time.perf_counter() 
    logger.info(f"Processing question: {q.question}")
    try:
        # Connect to existing Pinecone index
        vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        sources = retriever.invoke(q.question)
        
        context_text = "\n\n---\n\n".join([d.page_content for d in sources])
        
        prompt = PromptTemplate(
            template="Answer based ONLY on this context:\n{context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"]
        )
        
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY)
        chain = prompt | llm
        raw_response = chain.invoke({"context": context_text, "question": q.question})
        
        # Sanitize Gemini's response to extract ONLY text, ignoring internal signatures
        if isinstance(raw_response.content, list):
            answer_text = "".join([part.get("text", "") for part in raw_response.content if isinstance(part, dict) and part.get("type") == "text"])
        else:
            answer_text = raw_response.content

        # Defensive check: log a warning if the LLM returned an empty answer
        if not answer_text.strip():
            logger.warning(f"Sanitization resulted in empty answer for question: {q.question}")

        # Log latency for internal monitoring ONLY
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Query processed in {elapsed_time:.4f} seconds")
        
        # Return response (NO internal timing exposed to the client)
        return {
            "answer": answer_text,
            "sources": [{"page": d.metadata.get("page", "?"), "content": d.page_content[:200]} for d in sources]
        }
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
