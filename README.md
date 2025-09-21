# 🚀 Automated Resume Relevance Check System

> **AI-Powered Resume Evaluation Engine for Innomatics Research Labs**  
> *Scalable, consistent, and intelligent candidate screening with automated scoring and personalized feedback*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AI Powered](https://img.shields.io/badge/AI-Powered-purple.svg)](https://openai.com)

## 📋 Problem Statement

At **Innomatics Research Labs**, resume evaluation is currently manual, inconsistent, and time-consuming. Every week, the placement team across **Hyderabad, Bangalore, Pune, and Delhi NCR** receives **18–20 job requirements**, with each posting attracting thousands of applications.

Currently, recruiters and mentors manually review resumes, matching them against job descriptions (JD). This leads to:

- ⏱️ **Delays in shortlisting candidates**
- 🔄 **Inconsistent judgments**, as evaluators may interpret role requirements differently  
- 📊 **High workload for placement staff**, reducing their ability to focus on interview prep and student guidance
- 🏢 **Hiring companies expecting fast and high-quality shortlists**

With the growing scale of operations, there is a **pressing need for an automated system** that can scale, be consistent, and provide actionable feedback to students.

## 🎯 Objective

The **Automated Resume Relevance Check System** will:

- ⚙️ **Automate resume evaluation** against job requirements at scale
- 📊 **Generate a Relevance Score (0–100)** for each resume per job role
- 🔍 **Highlight gaps** such as missing skills, certifications, or projects
- ✅ **Provide a fit verdict** (High / Medium / Low suitability) to recruiters
- 💡 **Offer personalized improvement feedback** to students
- 🌐 **Store evaluations** in a web-based dashboard accessible to the placement team

This system should be **robust, scalable, and flexible** enough to handle thousands of resumes weekly.

## 🏗️ Proposed Solution

We propose building an **AI-powered resume evaluation engine** that combines rule-based checks with LLM-based semantic understanding.

The system will:

- 📄 **Accept resumes** (PDF/DOCX) uploaded by students
- 📋 **Accept job descriptions** uploaded by the placement team  
- 🔗 **Use text extraction + embeddings** to compare resume content with job descriptions
- ⚖️ **Run hybrid scoring:**
  - **Hard match** (keywords, skills, education)
  - **Soft match** (semantic fit via embeddings + LLM reasoning)
- 📈 **Output a Relevance Score, Missing Elements, and Verdict**
- 💾 **Store results** for the placement team in a searchable web application dashboard

This approach ensures both **speed** (hard checks) and **contextual understanding** (LLM-powered checks).

## 🔄 Workflow

### 1. **Job Requirement Upload** 
   - Placement team uploads job description (JD)

### 2. **Resume Upload**
   - Students upload resumes while applying

### 3. **Resume Parsing**
   - Extract raw text from PDF/DOCX
   - Standardize formats (remove headers/footers, normalize sections)

### 4. **JD Parsing** 
   - Extract role title, must-have skills, good-to-have skills, qualifications

### 5. **Relevance Analysis**
   - **Step 1:** Hard Match – keyword & skill check (exact and fuzzy matches)
   - **Step 2:** Semantic Match – embedding similarity between resume and JD using LLMs
   - **Step 3:** Scoring & Verdict – Weighted scoring formula for final score

### 6. **Output Generation**
   - Relevance Score (0–100)
   - Missing Skills/Projects/Certifications
   - Verdict (High / Medium / Low suitability)
   - Suggestions for student improvement

### 7. **Storage & Access**
   - Results stored in the database
   - The placement team can search/filter resumes by job role, score, and location

### 8. **Web Application**
   - Placement team dashboard: upload JD, see shortlisted resumes

## 🛠️ Tech Stack

### **Core Resume Parsing, AI Framework and Scoring Mechanism**

- **Python** – Primary programming language
- **PyMuPDF / pdfplumber** – Extract text from PDFs
- **python-docx / docx2txt** – Extract text from DOCX
- **spaCy / NLTK** – Entity extraction, text normalization
- **LangChain** – Orchestration of LLM workflows
- **LangGraph** – Structured stateful pipelines for resume–JD analysis
- **LangSmith** – Observability, testing, and debugging of LLM chains
- **Vector Store (Chroma / FAISS / Pinecone)** – For embeddings and semantic search
- **LLM Models** – OpenAI GPT / Gemini / Claude / HuggingFace models for semantic matching & feedback generation
- **Keyword Matching** – TF-IDF, BM25, fuzzy matching
- **Semantic Matching** – Embeddings + cosine similarity
- **Weighted Score** – Combine hard and soft matches into a final score

### **Web Application Stack**

- **Flask / FastAPI** – Backend APIs to process uploads, run evaluation, and serve results
- **Streamlit (MVP)** – Frontend for evaluators (upload, dashboard, review)
- **SQLite / PostgreSQL** – Database for storing results, metadata, and audit logs

## 📦 Installation Steps

### **Prerequisites**
- Python 3.11+
- Git
- (Optional) Docker for containerized deployment

### **1. Clone Repository**
```bash
git clone https://github.com/Naveenkm007/Automated-Resume-Relevance-Check-System.git
cd Automated-Resume-Relevance-Check-System
```

### **2. Environment Setup**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### **3. Install Dependencies**
```bash
# Install required packages
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### **4. Environment Configuration**
```bash
# Copy environment template
cp .env.example .env
# Edit .env with your API keys (optional for basic functionality)
```

### **5. Initialize Database (Optional)**
```bash
# For full API functionality
python -m api.init_db
```

## 🚀 Usage

### **Quick Start - Demo Mode**
```bash
# Run simple parsing demo
python simple_demo.py

# Run full feature demo
python demo.py

# Test scoring system
python simple_scoring_engine.py
```

### **Web Dashboard**
```bash
# Launch Streamlit dashboard
python -m streamlit run simple_dashboard.py
# Access at: http://localhost:8501

# Launch AI Insights Dashboard (Advanced)
python -m streamlit run ai_insights_dashboard.py
# Access at: http://localhost:8502
```

### **API Server (Production)**
```bash
# Start FastAPI backend
uvicorn api.main:app --reload
# API docs at: http://localhost:8000/docs

# Start background workers (new terminal)
celery -A api.tasks worker --loglevel=info
```

### **Docker Deployment**
```bash
# One-command setup
docker-compose up --build

# Initialize database
docker-compose exec api python -m api.init_db
```

### **Command Line Usage**
```bash
# Parse a single resume
python parse_sample.py path/to/resume.pdf

# Get human-readable summary
python parse_sample.py path/to/resume.pdf --format summary

# Save output to file
python parse_sample.py path/to/resume.pdf --output results.json
```

## 📊 Sample Data

For testing and demonstration purposes, sample data is available in the repository:
- **Sample Resume**: `sample_resume.txt` - Example resume text for testing
- **Test Cases**: Located in `tests/` directory
- **Example Outputs**: JSON formatted parsing results

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all validation tests
python validate_system.py

# Test resume parsing
pytest tests/test_parser.py -v

# Test scoring system
python validate_hard_match.py

# Test with coverage
pytest tests/ --cov=resume_parser --cov-report=html
```

## 🏗️ Architecture

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Resume Parser     │───▶│  Scoring Engine  │───▶│  Streamlit Dashboard│
│   (PDF/DOCX → JSON) │    │  (Hard + Semantic)│    │  (Interactive UI)   │
└─────────────────────┘    └──────────────────┘    └─────────────────────┘
           │                          │                          │
           ▼                          ▼                          ▼
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   spaCy NER         │    │   TF-IDF + AI    │    │   FastAPI Backend   │
│   (Entity Extract)  │    │   (Embeddings)   │    │   (Production API)  │
└─────────────────────┘    └──────────────────┘    └─────────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built for Innomatics Research Labs** | **AI-Powered Resume Intelligence** | **Scale • Consistency • Automation**
