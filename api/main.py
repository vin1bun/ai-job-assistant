import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import run_pipeline

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="AI Job Application Assistant",
    description="Multi-agent system for resume gap analysis",
    version="1.0.0"
)

# CORS — Streamlit frontend ke liye zaroori
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Health Check ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "message": "AI Job Assistant API is live!"}


# ── Main Endpoint ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    jd_text: str = Form(...),
    resume: UploadFile = File(...)
):
    """
    Main endpoint — accepts JD text + Resume PDF
    Returns full gap analysis
    """
    # Step 1 — Save uploaded PDF temporarily
    temp_path = f"temp_{resume.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)

    try:
        # Step 2 — Run full pipeline
        result = run_pipeline(jd_text, temp_path)

        # Step 3 — Extract outputs
        gap = result.get("gap_analysis", {})
        jd_analysis = result.get("jd_analysis", "")
        resume_analysis = result.get("resume_analysis", "")
        error = result.get("error", "")

        if error:
            return {"success": False, "error": error}

        return {
            "success": True,
            "jd_analysis": jd_analysis,
            "resume_analysis": resume_analysis,
            "gap_analysis": {
                "missing_skills": gap.get("MISSING_SKILLS", []),
                "present_skills": gap.get("PRESENT_SKILLS", []),
                "ats_keywords_missing": gap.get("ATS_KEYWORDS_MISSING", []),
                "weak_bullets": gap.get("WEAK_BULLETS_TO_IMPROVE", []),
                "new_bullets": gap.get("NEW_ATS_OPTIMIZED_BULLETS", []),
                "match_score": gap.get("MATCH_SCORE", ""),
                "priority_actions": gap.get("PRIORITY_ACTION_ITEMS", [])
            }
        }

    finally:
        # Step 4 — Temp file cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)