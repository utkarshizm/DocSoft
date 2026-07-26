import os
import logging
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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
    # In a real app, you might exit here, but we'll let it ride for the demo

PERSIST_DIR = "./chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)

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

        # Embed & Store (persisting to local disk)
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
        vectorstore.persist()
        
        os.unlink(tmp_path) # cleanup
        logger.info(f"Successfully indexed {len(chunks)} chunks from {file.filename}")
        return {"message": "Document indexed successfully", "chunks": len(chunks)}
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(q: Question):
    logger.info(f"Processing question: {q.question}")
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        sources = retriever.invoke(q.question)
        
        context_text = "\n\n---\n\n".join([d.page_content for d in sources])
        
        prompt = PromptTemplate(
            template="Answer based ONLY on this context:\n{context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"]
        )
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        chain = prompt | llm
        response = chain.invoke({"context": context_text, "question": q.question})
        
        logger.info("Successfully generated answer.")
        return {
            "answer": response.content,
            "sources": [{"page": d.metadata.get("page", "?"), "content": d.page_content[:200]} for d in sources]
        }
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
