# 📄 DocSoft — AI Document Q&A (RAG Pipeline)

A retrieval-augmented generation (RAG) app: upload PDFs, ask questions in plain English, and get answers grounded in the source text with page-level citations. No hallucinated answers — every claim traces back to a real chunk of the document.

**Stack:** LangChain · ChromaDB · Google Gemini (embeddings + LLM) · Streamlit

---

## Table of Contents

- [Features](#features)
- [Get a Free Gemini API Key](#1-get-a-free-gemini-api-key)
- [Run It Locally](#2-run-it-locally)
- [Suggested Documents to Demo With](#3-suggested-documents-to-demo-with)
- [Run the Retrieval Evaluation](#4-run-the-retrieval-evaluation-for-your-resume-metric)
- [Deploy It Live](#5-deploy-it-live-free)
- [Security Note](#security-note)

---

## Features

- 📤 Upload one or more PDFs and build a searchable knowledge base
- 💬 Ask questions in plain English, grounded strictly in the uploaded source text
- 🔗 Page-level citations for every answer — no hallucinations
- 🔀 Cross-document Q&A when multiple PDFs are loaded at once
- 📊 Built-in retrieval evaluation script to measure real accuracy

---

## 1. Get a Free Gemini API Key

Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and generate a key. It's free within generous rate limits — no credit card required.

## 2. Run It Locally

```bash
cd rag_app
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).

1. Paste your API key in the sidebar.
2. Upload a PDF (try a 10-K, an annual report, or your own project write-up).
3. Click **Build knowledge base**.
4. Ask questions.

## 3. Suggested Documents to Demo With

- A public company 10-K or annual report ([SEC EDGAR](https://www.sec.gov/edgar) has these free)
- Your own project reports (e.g. Olist / Zomato / IPL), exported as PDF
- A research paper from [arXiv](https://arxiv.org)

> 💡 Having 2–3 PDFs loaded at once and asking cross-document questions makes for a stronger demo than a single file.

## 4. Run the Retrieval Evaluation (for Your Resume Metric)

1. Open `eval_questions.json` and replace the placeholder questions with real questions about your test PDF, along with the page number (0-indexed, matching what `PyPDFLoader` reports) where the answer lives. Aim for 10–15 questions.

2. Set your API key as an environment variable:

   ```bash
   # Mac/Linux
   export GOOGLE_API_KEY="your-key-here"

   # Windows (cmd)
   set GOOGLE_API_KEY=your-key-here
   ```

3. Run the evaluation:

   ```bash
   python eval.py path/to/your.pdf eval_questions.json
   ```

4. It prints a hit/miss per question and a final accuracy, e.g.:

   ```
   Top-3 retrieval accuracy: 86.7%
   ```

   Use that real number in your resume bullet — don't guess or estimate it.

## 5. Deploy It Live (Free)

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and deploy the repo (point it at `app.py`).
3. You'll get a live link like: `your-app-name.streamlit.app` — add it to your resume the same way you did for your other Streamlit projects.

## Security Note

⚠️ **Don't hardcode your API key anywhere in the repo.** The app takes it as user input in the sidebar, so each visitor (including recruiters trying it out) enters their own key. This is the standard, safe pattern for public Streamlit deployments.

---

## Project Structure

```
rag_app/
├── app.py                 # Streamlit app entry point
├── eval.py                 # Retrieval evaluation script
├── eval_questions.json     # Your test questions + answer page numbers
└── requirements.txt        # Python dependencies
```

## License

MIT
