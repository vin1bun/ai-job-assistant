🤖 AI Job Assistant — Backend

> A multi-agent AI system that analyzes job descriptions against your resume and generates ATS-optimized gap analysis using LangGraph orchestration.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3.3--70B-purple)](https://groq.com)

---

## 🧠 What This Does

Paste a Job Description + upload your Resume PDF → the system runs 3 specialized AI agents in sequence → returns a detailed gap analysis with match score, missing skills, and ATS-optimized bullet points.

**End-to-end tested: 70/100 match score working ✅**

---

## 🏗️ System Architecture
User Input (JD + Resume PDF)
↓
FastAPI Backend
↓
LangGraph Orchestrator
┌────────────────────────────────────┐
│                                    │
▼                                    │
JD Agent          Resume Agent      Gap Agent
(Extracts          (Parses PDF       (Compares both,
skills, ATS        via FAISS         generates match
keywords,          vector store)     score + bullets)
requirements)
│                    │                  │
└────────────────────┴──────────────────┘
↓
JSON Response to Frontend

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| Agent Logic | LangChain |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Vector Store | FAISS |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| PDF Parsing | pdfplumber |
| Language | Python 3.11 |

---

## 🤖 Agents

### 1. JD Agent
- Parses job description using LLM
- Extracts: required skills, nice-to-have skills, ATS keywords, experience level, red flags

### 2. Resume Agent
- Extracts raw text from resume PDF using pdfplumber
- Chunks text → stores in FAISS vector store
- LLM extracts: candidate name, skills, experience, education, projects, certifications

### 3. Gap Agent
- Receives outputs from both agents
- Returns: missing skills, present skills, ATS keywords missing, weak bullets, 5–7 new ATS-optimized bullets, match score out of 100, top 3 priority action items

---

## 📁 Folder Structure

ai-job-assistant/
├── agents/
│   ├── jd_agent.py
│   ├── resume_agent.py
│   └── gap_agent.py
├── api/
│   └── main.py
├── graph/
│   └── workflow.py
├── outputs/
└── .env

---

## 🚀 Local Setup

# 1. Clone the repo
git clone https://github.com/vin1bun/ai-job-assistant.git
cd ai-job-assistant

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
echo GROQ_API_KEY=your_groq_api_key_here > .env

# 5. Run the server
python api/main.py

Server runs at: http://localhost:8000

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /analyze | Analyze JD vs Resume |

POST /analyze accepts multipart/form-data:
- jd_text (string) — Job description
- resume (file) — Resume PDF

---

## 🔗 Related

- Frontend Repo: https://github.com/vin1bun/ai-job-frontend
- Portfolio: https://vin1bun.github.io
- LinkedIn: https://linkedin.com/in/vineetprakash03

---

## 👨‍💻 Author

Vineet Prakash — AI/ML Engineer & Data Scientist
- vineetprakash03@gmail.com
- https://github.com/vin1bun
