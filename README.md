# 🤖 HR Knowledge Assistant

> **A production-ready RAG pipeline that turns HR policy documents into an intelligent Q&A assistant — powered by LLaMA 3, Groq, LangChain, and FAISS.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=flat)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-0467DF?style=flat)](https://faiss.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)

---
TRY IT HERE : [ STREAMLIT](https://hr-knowledge-assistant.streamlit.app/)

## 🚀 What It Does

HR teams drown in policy PDFs and employee handbooks. This assistant lets anyone ask natural language questions and get **accurate, grounded answers** directly from internal HR documents — no hallucinations, no Googling, no waiting.

Built on a **Retrieval-Augmented Generation (RAG)** architecture, it retrieves the most relevant document chunks and feeds them as context to **LLaMA 3 via Groq's blazing-fast LPU inference** before generating a response.

---

## 🏗️ Architecture

```
HR Documents (PDFs)
        │
        ▼
 [Text Extraction]             ← pypdf
        │
        ▼
 [Text Chunking]               ← LangChain Text Splitters
        │
        ▼
 [Embedding Generation]        ← HuggingFace sentence-transformers
        │
        ▼
 [FAISS Vector Store]          ← Indexed & persisted locally
        │
   User Query
        │
        ▼
 [HyDE Expansion] ─ToggleON/OFF─ LLaMA 3.3-70b generates a hypothetical
        │                          HR policy excerpt matching the query
        │                          (falls back to raw query if disabled/fails)
        ▼
 [Semantic Retrieval]          ← Top-k similarity search on hypothetical doc
        │
        ▼
 [LLM Generation]              ← LLaMA 3 via Groq API
        │
        ▼
 [Streamlit UI]                ← Chat interface + API key input + HyDE toggle
```

---

## ✨ Key Features

- **HyDE (Hypothetical Document Embeddings)** — Before retrieval, LLaMA 3.3-70b generates a fake HR policy excerpt that *would* answer the query. This hypothetical doc is embedded and used for retrieval instead of the raw query — bridging the semantic gap between short questions and long policy documents. Gracefully falls back to the original query on failure.
- **Toggleable HyDE** — HyDE can be switched on or off directly in the UI, so you can compare retrieval quality with and without it in real time
- **In-app API Key Input** — Groq API key is entered inside the Streamlit app itself (no `.env` file needed), making it straightforward to deploy and share without exposing credentials
- **RAG Pipeline** — Retrieves only the most relevant document chunks before generating answers, keeping responses grounded and accurate
- **Groq LPU Inference** — Ultra-low latency responses using Groq's hardware-accelerated LLaMA 3
- **Local Vector Search** — FAISS index stored locally, no external vector DB dependency
- **Modular Codebase** — Clean separation between text extraction, embedding, and the main app
- **Streamlit Chat UI** — Conversational interface with session history

---

## 🗂️ Project Structure

```
HR-Knowledge-Assistant/
│
├── text_extract/          # PDF ingestion & text parsing (pypdf)
├── embedding/             # Chunking + HuggingFace embedding + FAISS index
├── models/                # LLM chain configuration (LangChain + Groq)
├── main_app/              # Streamlit chat interface
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | LLaMA 3 (via Groq API) |
| Framework | LangChain |
| Embeddings | HuggingFace `sentence-transformers` |
| Vector Store | FAISS (CPU) |
| PDF Parsing | pypdf |
| UI | Streamlit |
| Language | Python 3.12+ |

---

## 🛠️ Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/MLbyTharun/HR-Knowledge-Assistant.git
cd HR-Knowledge-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run main_app/app.py
```

### 4. Add your HR documents & build the vector index

Place your HR policy PDFs in the designated input folder (see `text_extract/`), then run:

```bash
python embedding/embed.py
```

### 5. Enter your Groq API key in the app

No `.env` file needed — paste your Groq API key directly into the sidebar of the running app.

> Get a free API key at [console.groq.com](https://console.groq.com)

### 6. Toggle HyDE on or off

Use the **HyDE toggle** in the UI to enable or disable Hypothetical Document Embedding per query. Useful for comparing retrieval quality with and without it.

---

## 🧠 How HyDE Works

Standard RAG embeds the **user's short query** and searches for similar document chunks — but HR policy docs are long and dense, making this query-to-document match imprecise.

With HyDE enabled, the pipeline does this instead:

```
User Query: "What is the leave policy for new employees?"
        │
        ▼
LLaMA 3.3-70b generates a hypothetical policy excerpt:
  "New employees are entitled to 12 days of paid leave per calendar year,
   accruing at 1 day per month. Leave may not be carried forward beyond..."
        │
        ▼
Embed the hypothetical doc → search FAISS
        │
        ▼
Now doing document-to-document similarity (much better match)
```

The LLM runs at `temperature=0.1` to keep the hypothetical grounded. If generation fails for any reason, it silently falls back to the original query — so the app never breaks.

---



- **HR Chatbots** — Employee self-service for policy questions
- **Onboarding Assistants** — Help new hires navigate handbooks instantly
- **Compliance Q&A** — Quickly surface relevant policy clauses
- **Internal Knowledge Bases** — Extend to any domain-specific document corpus

---

## 🔭 Roadmap

- [ ] Multi-document support with source citation
- [ ] Conversation memory across sessions
- [ ] Reranking layer (Cohere / cross-encoder)
- [ ] Docker containerization for one-command deploy
- [ ] REST API wrapper (FastAPI) for integration

---

## 🤝 Contributing

PRs and issues welcome! If you extend this to a new domain (legal, finance, product docs), open a PR — would love to see it.

---

## 📬 Contact

Built by **Tharun** · [GitHub](https://github.com/MLbyTharun)

---

*If this project helped you, drop a ⭐ — it helps others find it!*