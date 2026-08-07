import os
import httpx
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# --- CONFIG & SECURITY ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DOCSOFT_API_KEY = os.getenv("DOCSOFT_API_KEY")

if not GEMINI_API_KEY or not DOCSOFT_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY or DOCSOFT_API_KEY in environment variables.")

API_URL = "https://docsoft-tueu.onrender.com/ask"
HEADERS = {
    "Authorization": f"Bearer {DOCSOFT_API_KEY}",
    "Content-Type": "application/json"
}

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

EVAL_QUESTIONS = [
    {"q": "What university did I attend?", "a": "JSS Academy"},
    {"q": "What was the outcome of the Olist project?", "a": "900%"},
    {"q": "How many students did I mentor at Smile Foundation?", "a": "20"},
    {"q": "What was my role at OdNest?", "a": "Tech Community Lead"},
    {"q": "What tools did I use for the IPL dashboard?", "a": "Streamlit"},
    {"q": "Did I complete a simulation for JP Morgan?", "a": "Yes"},
    {"q": "What programming language is commonly used for data science?", "a": "Python"},
    {"q": "Is Docker used for containerization?", "a": "Yes"},
    {"q": "What is LangChain used for?", "a": "AI"},
]

async def ask_rag(client, question):
    payload = {"question": question}
    try:
        response = await client.post(API_URL, headers=HEADERS, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json().get("answer", "ERROR: No answer field")
    except Exception as e:
        return f"ERROR: {e}"

def ask_baseline(question):
    prompt = f"Answer this question concisely in one sentence: {question}"
    raw_response = llm.invoke([HumanMessage(content=prompt)])
    
    # Sanitize Gemini's list response
    if isinstance(raw_response.content, list):
        return "".join([part.get("text", "") for part in raw_response.content if isinstance(part, dict) and part.get("type") == "text"])
    return raw_response.content

def grade_answer(expected, actual):
    return expected.lower() in actual.lower()

async def main():
    print("Starting Evaluation...\n")
    rag_correct = 0
    base_correct = 0
    
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(EVAL_QUESTIONS):
            q = item["q"]
            expected = item["a"]
            
            rag_ans = await ask_rag(client, q)
            rag_grade = grade_answer(expected, rag_ans)
            if rag_grade:
                rag_correct += 1
            
            base_ans = ask_baseline(q)
            base_grade = grade_answer(expected, base_ans)
            if base_grade:
                base_correct += 1
                
            print(f"Q{i+1}: {q}")
            print(f"  Baseline: {'CORRECT' if base_grade else 'INCORRECT'} | RAG: {'CORRECT' if rag_grade else 'INCORRECT'}")
            await asyncio.sleep(1)
            
    total = len(EVAL_QUESTIONS)
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Baseline (No Context) Accuracy: {base_correct}/{total} ({(base_correct/total)*100:.0f}%)")
    print(f"DocSoft (RAG) Accuracy:         {rag_correct}/{total} ({(rag_correct/total)*100:.0f}%)")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
