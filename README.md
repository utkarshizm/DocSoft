# AI Document Q&A — RAG Pipeline

A retrieval-augmented generation (RAG) app: upload PDFs, ask questions in plain
English, get answers grounded in the source text with page-level citations —
no hallucinated answers, every claim traces back to a real chunk of the document.

**Stack:** LangChain · ChromaDB · Google Gemini (embeddings + LLM) · Streamlit

---

## 1. Get a free Gemini API key

Go to https://aistudio.google.com/app/apikey and generate a key. It's free
within generous rate limits — no credit card needed.

## 2. Run it locally

```bash
cd rag_app
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser (usually http://localhost:8501).
Paste your API key in the sidebar, upload a PDF (try a 10-K, an annual
report, or your own Olist/Zomato project write-up), click
**Build knowledge base**, then ask questions.

## 3. Suggested documents to demo with

- A public company 10-K or annual report (SEC EDGAR has these free)
- Your own Olist / Zomato / IPL project reports, exported as PDF
- A research paper (arXiv)

Having 2–3 PDFs loaded at once and asking cross-document questions makes
for a stronger demo than a single file.

## 4. Run the retrieval evaluation (for your resume metric)

1. Open `eval_questions.json` and replace the placeholder questions with
   real questions about your test PDF, along with the page number
   (0-indexed, matching what PyPDFLoader reports) where the answer lives.
   Aim for 10-15 questions.
2. Set your API key as an environment variable:
   ```bash
   export GOOGLE_API_KEY="your-key-here"      # Mac/Linux
   set GOOGLE_API_KEY=your-key-here           # Windows (cmd)
   ```
3. Run:
   ```bash
   python eval.py path/to/your.pdf eval_questions.json
   ```
4. It prints a hit/miss per question and a final accuracy, e.g.
   `Top-3 retrieval accuracy: 86.7%`. Use that real number in your resume
   bullet — don't guess or estimate it.

## 5. Deploy it live (free)

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, and deploy the repo
   (point it at `app.py`).
3. You'll get a live link like your other Streamlit projects
   (`your-app-name.streamlit.app`) — add it to your resume the same way you
   did for Olist and IPL.

**Note:** don't hardcode your API key anywhere in the repo. The app takes it
as a user input in the sidebar, so each visitor (including recruiters trying
it out) enters their own key — this is the standard, safe pattern for public
Streamlit deployments.



---


