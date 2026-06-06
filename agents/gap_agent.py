import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

gap_prompt = PromptTemplate(
    input_variables=["jd_analysis", "resume_analysis"],
    template="""
You are an expert career coach and ATS optimization specialist.

You are given:
1. JD Analysis (from Job Description Agent)
2. Resume Analysis (from Resume Parser Agent)

Your task is to perform a thorough skill gap analysis and generate actionable output.

--- JD ANALYSIS ---
{jd_analysis}

--- RESUME ANALYSIS ---
{resume_analysis}

--- YOUR OUTPUT FORMAT (strictly follow this) ---

MISSING_SKILLS:
- List skills mentioned in JD but completely absent in resume

PRESENT_SKILLS:
- List skills that match between JD and resume

ATS_KEYWORDS_MISSING:
- List ATS keywords from JD that are not found in resume

WEAK_BULLETS_TO_IMPROVE:
- Identify existing resume bullets that are vague or lack impact

NEW_ATS_OPTIMIZED_BULLETS:
- Write 5-7 strong, ATS-friendly resume bullet points
- Use action verbs + metrics where possible
- Incorporate missing ATS keywords naturally

MATCH_SCORE:
- Give an overall resume-JD match score out of 100
- Brief explanation

PRIORITY_ACTION_ITEMS:
- Top 3 things the candidate should do immediately
"""
)

def analyze_gap(jd_analysis: str, resume_analysis: str) -> dict:
    print("\n[Gap Agent] Analyzing skill gap...\n")
    
    # Modern LangChain syntax — no LLMChain needed
    chain = gap_prompt | llm
    response = chain.invoke({
        "jd_analysis": jd_analysis,
        "resume_analysis": resume_analysis
    })
    
    raw_output = response.content
    result = parse_gap_output(raw_output)
    return result


def parse_gap_output(raw_text: str) -> dict:
    sections = {
        "MISSING_SKILLS": [],
        "PRESENT_SKILLS": [],
        "ATS_KEYWORDS_MISSING": [],
        "WEAK_BULLETS_TO_IMPROVE": [],
        "NEW_ATS_OPTIMIZED_BULLETS": [],
        "MATCH_SCORE": "",
        "PRIORITY_ACTION_ITEMS": []
    }

    current_section = None

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        matched = False
        for key in sections.keys():
            if line.startswith(key):
                current_section = key
                matched = True
                break

        if not matched and current_section:
            clean_line = line.lstrip("-•* ").strip()
            if not clean_line:
                continue
            if isinstance(sections[current_section], list):
                sections[current_section].append(clean_line)
            else:
                sections[current_section] += clean_line + " "

    return sections


if __name__ == "__main__":
    sample_jd = """
    REQUIRED_SKILLS: Python, SQL, Machine Learning, XGBoost, LangChain, FastAPI, Docker
    NICE_TO_HAVE_SKILLS: Kubernetes, Spark, Airflow
    EXPERIENCE_LEVEL: 2-4 years
    ATS_KEYWORDS: NLP, RAG, Vector Database, LLM, REST API, CI/CD, Model Deployment
    JOB_TITLE: ML Engineer
    JOB_SUMMARY: Build and deploy ML models and LLM-based applications.
    RED_FLAGS: Requires Docker experience mandatory
    """

    sample_resume = """
    CANDIDATE_NAME: Vineet Prakash
    SKILLS: Python, SQL, XGBoost, SHAP, Pandas, NumPy, LangChain, FAISS, Streamlit, Groq
    EXPERIENCE_YEARS: Fresher (0-1 year)
    EDUCATION: B.E. Electronics & Communication, Dayananda Sagar College 2021
    PROJECTS: AskVineet RAG Chatbot, Sales Forecasting (Prophet+XGBoost), Customer Churn, Loan Default Prediction
    CERTIFICATIONS: IBM Data Science Professional, Google Prompting Essentials, Microsoft Generative AI
    """

    result = analyze_gap(sample_jd, sample_resume)

    print("=" * 60)
    print("GAP ANALYSIS RESULT")
    print("=" * 60)
    for section, content in result.items():
        print(f"\n{section}:")
        if isinstance(content, list):
            for item in content:
                print(f"  - {item}")
        else:
            print(f"  {content}")
