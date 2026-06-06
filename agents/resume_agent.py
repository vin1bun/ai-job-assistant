from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pdfplumber
import os


def extract_resume_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def parse_resume(pdf_path: str) -> dict:
    
    raw_text = extract_resume_text(pdf_path)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(raw_text)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert resume analyzer.
        Extract the following from the resume in this EXACT format:
        
        CANDIDATE_NAME: [name]
        SKILLS: [skill1, skill2, skill3]
        EXPERIENCE_YEARS: [number]
        EDUCATION: [degree, college]
        PROJECTS: [project1, project2]
        CERTIFICATIONS: [cert1, cert2]
        """),
        ("human", "Analyze this resume: {resume_text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"resume_text": raw_text[:3000]})
    
    return {
        "raw_text": raw_text,
        "chunks": chunks,
        "vector_store": vector_store,
        "parsed_output": response.content
    }


if __name__ == "__main__":
    result = parse_resume("resume.pdf")
    print(result["parsed_output"])
