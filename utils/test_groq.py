from langchain_groq import ChatGroq
import os

os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

response = llm.invoke("Say hello in one sentence")
print(response.content)
