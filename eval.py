"""
Retrieval evaluation script.
-----------------------------
Measures top-k retrieval accuracy: for each test question, does the correct
source page show up in the top-k retrieved chunks?

This gives you the metric for your resume bullet, e.g.:
"Evaluated retrieval quality on a custom 15-question test set, achieving
82% top-3 retrieval accuracy."

Usage:
    1. Fill in eval_questions.json with real questions + the page number
       (from your PDF) that contains the answer.
    2. Set GOOGLE_API_KEY as an environment variable.
    3. Run: python eval.py path/to/your.pdf
"""

import os
import sys
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_index(pdf_path, chunk_size=1000, chunk_overlap=150):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    return vectorstore


def evaluate(vectorstore, questions, k=3):
    correct = 0
    results = []
    for item in questions:
        q = item["question"]
        expected_page = item["expected_page"]

        retrieved = vectorstore.similarity_search(q, k=k)
        retrieved_pages = [doc.metadata.get("page") for doc in retrieved]

        hit = expected_page in retrieved_pages
        correct += int(hit)
        results.append({
            "question": q,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "hit": hit,
        })

    accuracy = correct / len(questions) if questions else 0
    return accuracy, results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval.py path/to/your.pdf [path/to/eval_questions.json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    questions_path = sys.argv[2] if len(sys.argv) > 2 else "eval_questions.json"

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Set GOOGLE_API_KEY as an environment variable first.")
        sys.exit(1)

    with open(questions_path) as f:
        questions = json.load(f)

    print(f"Building index from {pdf_path} ...")
    vectorstore = build_index(pdf_path)

    K = 5
    print(f"Evaluating top-{K} retrieval accuracy on {len(questions)} questions ...")
    accuracy, results = evaluate(vectorstore, questions, k=K)

    for r in results:
        status = "✅" if r["hit"] else "❌"
        print(f"{status} Q: {r['question']}")
        print(f"    expected page {r['expected_page']}, retrieved pages {r['retrieved_pages']}")

    print(f"\nTop-{K} retrieval accuracy: {accuracy * 100:.1f}%")
