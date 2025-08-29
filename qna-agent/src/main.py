# src/main.py
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun


# --- 1. SET-UP LLM --------
def get_llm():
    """
    Initialize and return a ChatOpenAI object with specific configurations.
    """
    return ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="not-needed",
        model="gpt-oss-20b",
        temperature=0,  
    )
def load_metrics_from_json(file_path):
    """
    Load metrics data from a JSON file.
    """
    loader = JSONLoader(
        file_path=file_path, 
        jq_schema='.data.metrics[]', 
        text_content=False)
    return loader.load()

def create_vector_store(documents):
    """
    Create a vector store from a list of documents.
    """
    print("Creating embedding ...")
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
    )
    print("Creating FAISS vector store")
    return FAISS.from_documents(documents, embedding=embeddings)
def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    document_chain = create_stuff_documents_chain(
        llm=get_llm(), 
        prompt=prompt
        )
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
        )
    return retrieval_chain

def get_retrieval_chain():
    """
    Creates and returns a retrieval chain for querying metric definitions.
    This version is updated to handle the complex, nested JSON structure.
    """
    script_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(script_dir, "..", "data", "metrics.json")
    print("Loading metrics from JSON...")
    jq_schema = """
    .data.metrics[] | {
        "doc_content": (
            "Metric Name: \\(.name) | " +
            "Description: \\(.description) | " +
            "Calculation: \\(.function_operation) on key \\(.function_key) | " +
            "Metric Group: \\(.metric_group)"
        )
        }
        """
    loader = JSONLoader(
        file_path=json_file_path,
        jq_schema=jq_schema,
        text_content=False)  # Set text_content to True to load text content from JSON file_path=json_file_path, jq_schema=jq_schema, text_content=True
    documents  = loader.load()
     
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    document_chain = create_stuff_documents_chain(
        llm=get_llm(), 
        prompt=prompt
        )
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
        )
    return retrieval_chain

def setup_tools():
    """
    Set up and return a list of tools for the agent.
    """
    from langchain.agents import Tool

    retrieval_chain = get_retrieval_chain()
    
    def retrieval_func(query):
        result = retrieval_chain.invoke({"input": query})
        return result["answer"]
    
    retrival_tools = Tool(
            name="metric_definitions_search",
            func=retrieval_func,
            description="Useful for when you need to answer questions about metric definitions. Input should be a fully formed question.",
        )
    web_search_tool = DuckDuckGoSearchRun()
    tools = [retrival_tools, web_search_tool]
    return tools
def create_agent(tools):
    """
    Creates the ReAct agent.
    """
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=get_llm(),
        tools=tools,
        prompt=prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def main():
    """
    Main function to run the agent.
    """
    tools = setup_tools()
    agent_executor = create_agent(tools)
    print("Agent is ready! Ask me about metric definitions or anything else.")
    print("-" * 30)
    # Example query
    query = "Create a metric for std dev on amount usd for 1 weeks on payment score?"
    print(f"Query: {query}")
    
    response = agent_executor.invoke({"input": query})
    print(f"Response: {response['output']}")
    # --- Test Case 2: A question that requires web search ---
    question2 = "Who is the current CEO of Google?"
    print(f"QUESTION: {question2}")
    response2 = agent_executor.invoke({"input": question2})
    print(f"ANSWER: {response2['output']}")

if __name__ == "__main__":
    main()