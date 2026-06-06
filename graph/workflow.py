import os
import sys

# Path fix so agents can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.jd_agent import analyze_jd
from agents.resume_agent import parse_resume
from agents.gap_agent import analyze_gap


# ── 1. State Schema ─────────────────────────────────────────
# Yeh workflow ka "memory" hai — har agent isme read/write karta hai

class JobAppState(TypedDict):
    # Inputs
    jd_text: str
    resume_pdf_path: str

    # Agent outputs
    jd_analysis: str
    resume_analysis: str
    gap_analysis: dict

    # Status tracking
    current_step: str
    error: str


# ── 2. Node Functions ────────────────────────────────────────
# Har agent ek "node" hai LangGraph mein

def jd_node(state: JobAppState) -> JobAppState:
    """Node 1 — JD Analyzer"""
    print("\n[Workflow] Step 1: Running JD Agent...")
    try:
        result = analyze_jd(state["jd_text"])
        # Convert dict to readable string for next agent
        jd_str = "\n".join([f"{k}: {v}" for k, v in result.items()])
        return {
            **state,
            "jd_analysis": jd_str,
            "current_step": "jd_done"
        }
    except Exception as e:
        return {**state, "error": f"JD Agent failed: {str(e)}"}


def resume_node(state: JobAppState) -> JobAppState:
    """Node 2 — Resume Parser"""
    print("\n[Workflow] Step 2: Running Resume Agent...")
    try:
        result = parse_resume(state["resume_pdf_path"])
        # Convert dict to readable string for next agent
        resume_str = "\n".join([f"{k}: {v}" for k, v in result.items()])
        return {
            **state,
            "resume_analysis": resume_str,
            "current_step": "resume_done"
        }
    except Exception as e:
        return {**state, "error": f"Resume Agent failed: {str(e)}"}


def gap_node(state: JobAppState) -> JobAppState:
    """Node 3 — Skill Gap Analyzer"""
    print("\n[Workflow] Step 3: Running Gap Agent...")
    try:
        result = analyze_gap(state["jd_analysis"], state["resume_analysis"])
        return {
            **state,
            "gap_analysis": result,
            "current_step": "complete"
        }
    except Exception as e:
        return {**state, "error": f"Gap Agent failed: {str(e)}"}


# ── 3. Build the Graph ───────────────────────────────────────

def build_workflow():
    graph = StateGraph(JobAppState)

    # Add nodes
    graph.add_node("jd_agent", jd_node)
    graph.add_node("resume_agent", resume_node)
    graph.add_node("gap_agent", gap_node)

    # Define flow: jd → resume → gap → END
    graph.set_entry_point("jd_agent")
    graph.add_edge("jd_agent", "resume_agent")
    graph.add_edge("resume_agent", "gap_agent")
    graph.add_edge("gap_agent", END)

    return graph.compile()


# ── 4. Run Function (called by FastAPI / Streamlit later) ────

def run_pipeline(jd_text: str, resume_pdf_path: str) -> dict:
    """
    Main function to run the full pipeline.
    Returns final state with all agent outputs.
    """
    workflow = build_workflow()

    initial_state = {
        "jd_text": jd_text,
        "resume_pdf_path": resume_pdf_path,
        "jd_analysis": "",
        "resume_analysis": "",
        "gap_analysis": {},
        "current_step": "start",
        "error": ""
    }

    print("\n" + "="*60)
    print("AI JOB APPLICATION ASSISTANT — PIPELINE STARTED")
    print("="*60)

    final_state = workflow.invoke(initial_state)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    return final_state


# ── 5. Quick Test ────────────────────────────────────────────

if __name__ == "__main__":

    sample_jd = """
    We are looking for an ML Engineer with strong experience in Python, 
    SQL, Machine Learning, XGBoost, LangChain, FastAPI, and Docker.
    Nice to have: Kubernetes, Spark.
    Experience: 2-4 years.
    You will build and deploy ML models and LLM-based RAG applications.
    Must have experience with REST APIs, CI/CD pipelines, and Model Deployment.
    """

    # Use the resume.pdf already in project root
    resume_path = "resume.pdf"

    result = run_pipeline(sample_jd, resume_path)

    # Print final gap analysis
    print("\n\nFINAL GAP ANALYSIS:")
    print("="*60)
    gap = result.get("gap_analysis", {})
    for section, content in gap.items():
        print(f"\n{section}:")
        if isinstance(content, list):
            for item in content:
                print(f"  - {item}")
        else:
            print(f"  {content}")

    if result.get("error"):
        print(f"\n⚠️  ERROR: {result['error']}")