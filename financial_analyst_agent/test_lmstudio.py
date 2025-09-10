import os
os.environ["LMSTUDIO_MODEL"] = "openai/gpt-oss-20b"
os.environ["LMSTUDIO_BASE_URL"] = "http://localhost:1234/v1"

from langchain_openai import ChatOpenAI

# Test LM Studio connection
llm = ChatOpenAI(
    model="openai/gpt-oss-20b", 
    base_url="http://localhost:1234/v1", 
    api_key="lm-studio"
)

print("Model: openai/gpt-oss-20b")
print("Base URL: http://localhost:1234/v1")

try:
    response = llm.invoke("What is 2+2?")
    print("Success:", response.content)
except Exception as e:
    print("Error:", str(e))