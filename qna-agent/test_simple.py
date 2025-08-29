# Simple test without ReAct agent

import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_llm():
    return ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="not-needed",
        model="gpt-oss-20b", 
        temperature=0,
    )

def test_retrieval():
    script_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(script_dir, "data", "metrics.json")
    
    jq_schema = (
        '.data.metrics[] | "Metric Name: \\(.name) | '
        'Description: \\(.description) | '
        'Calculation: \\(.function_operation) on key \\(.function_key) | '
        'Metric Group: \\(.metric_group // "Not specified") | '
        'Status: \\(.status) | '
        'Duration: \\(.duration_value) \\(.duration_type)"'
    )
    
    print("Loading JSON data...")
    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema=jq_schema,
        text_content=False
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # Find and print the specific metric
    for doc in documents:
        if "cntd_cash_users_device_30d" in doc.page_content:
            print("\nFound the metric:")
            print(doc.page_content)
            break
    
    print("\nCreating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    retriever = vector_store.as_retriever()
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context.
        
        Context: {context}
        Question: {input}
        
        Answer:
        """
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("\nTesting retrieval...")
    question = "Tell me about the metric cntd_cash_users_device_30d"
    result = retrieval_chain.invoke({"input": question})
    print(f"Answer: {result['answer']}")
    
    print("\nTesting metric group question...")
    question2 = "What is the metric group for cntd_cash_users_device_30d?"
    result2 = retrieval_chain.invoke({"input": question2})
    print(f"Answer: {result2['answer']}")

if __name__ == "__main__":
    test_retrieval()