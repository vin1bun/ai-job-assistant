from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os


def analyze_jd(jd_text: str) -> dict:

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert ATS (Applicant Tracking System) analyst and recruiter with 10 years of experience.

        Analyze the job description and extract the following in this EXACT format:

        REQUIRED_SKILLS: [skill1, skill2, skill3]
        NICE_TO_HAVE_SKILLS: [skill1, skill2]
        EXPERIENCE_LEVEL: [fresher/junior/mid/senior]
        ATS_KEYWORDS: [keyword1, keyword2, keyword3]
        JOB_TITLE: [exact job title]
        JOB_SUMMARY: [one line summary]
        RED_FLAGS: [anything a candidate must not miss]

        Focus on ATS keywords that recruiters search for.
        Be specific and technical. Do not add extra explanation."""),
        ("human", "Analyze this job description for ATS optimization: {jd_text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"jd_text": jd_text})

    return {"raw_output": response.content}


if __name__ == "__main__":

    sample_jd = """
    We are looking for a Data Scientist with strong experience in Python,
    Machine Learning, and SQL. The candidate should have hands-on knowledge of
    TensorFlow, scikit-learn, pandas, and data visualization tools like Tableau or PowerBI.
    Experience with cloud platforms like AWS or GCP is a plus.
    1-2 years experience preferred. Must have strong communication skills.
    """

    result = analyze_jd(sample_jd)
    print(result["raw_output"])
