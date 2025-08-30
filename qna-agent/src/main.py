# src/main.py

import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import create_openai_tools_agent, AgentExecutor, Tool
import qdrant_client



os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. LLM AND TOOLS SETUP ---

def get_llm():
    """Initializes and returns the local LLM."""
    return ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="not-needed",
        # Note: model name is often not required for LM Studio but can be useful
        model="gpt-oss-20b", 
        temperature=0,
    )


def get_retrieval_chain(llm):
    """
    Creates the complete RAG chain for retrieving metric definitions.
    This function now handles loading, embedding, and chaining.
    """
    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = "metric-definitions"

    try:
        client.get_collection(collection_name)
        print("Collection already exists. Connecting to it...")
        vector_store = Qdrant(client, collection_name, embeddings)
    except Exception as e:
        print("Qdrant collection not found. Creating it and adding documents...")
        # If the collection doesn't exist, create it and add documents
        script_dir = os.path.dirname(__file__)
        json_file_path = os.path.join(script_dir, "..", "data", "metrics.json")
    
        # This jq schema extracts and formats the metric data with all relevant fields
        jq_schema = (
            '.data.metrics[] | "Metric Name: \\(.name) | '
            'Description: \\(.description) | '
            'Topic: \\(.topic) | '
            'Filter: \\(.filter // "Not specified") | '
            'Function_Operation: \\(.function_operation // "Not specified") | '
            'Global_Entity_IDs: \\(.global_entity_ids // "Not specified") | '
            'Calculation: \\(.function_operation) on key \\(.function_key) | '
            'Metric Group: \\(.metric_group // "Not specified") | '
            'Status: \\(.status) | '
            'Duration: \\(.duration_value) \\(.duration_type)"'
        )
    
        loader = JSONLoader(
            file_path=json_file_path,
            jq_schema=jq_schema,
            text_content=False 
        )
        documents = loader.load()
        vector_store = Qdrant.from_documents(
            documents,
            embeddings,
            host="localhost",
            port=6333,
            collection_name=collection_name,
        )
     
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that provides information about business metrics.
        Answer the following question based ONLY on the provided context.
        If the information is not in the context, say "I don't have information about that metric."
        
        <context>
        {context}
        </context>
        
        Question: {input}
        
        Answer:
        """
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def setup_tools(llm):
    """Sets up the tools the agent can use."""
    retrieval_chain = get_retrieval_chain(llm)
    
    def search_metrics(query):
        """Wrapper function to handle the retrieval chain properly."""
        try:
            if isinstance(query, dict):
                query = query.get('input', str(query))
            result = retrieval_chain.invoke({"input": query})
            return result.get('answer', 'No information found.')
        except Exception as e:
            return f"Error searching metrics: {str(e)}"
    
    retrieval_tool = Tool(
        name="metric_definitions_search",
        func=search_metrics,
        description="Search for internal business metric definitions. Use this to find information about specific metrics, their names, descriptions, calculations, or metric groups."
    )
    web_search_tool = DuckDuckGoSearchRun()
    return [retrieval_tool, web_search_tool]


# --- 2. AGENT CREATION ---

def create_agent(llm, tools):
    """Creates an agent that uses structured tool calling."""
    # 1. Pull a prompt designed for OpenAI Tools agents
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # 2. Use the modern 'create_openai_tools_agent'
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
        )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3 
    )
    return agent_executor

# --- 3. MAIN EXECUTION ---

def main():
    """Main function to initialize and run the agent."""
    llm = get_llm()
    tools = setup_tools(llm)
    agent_executor = create_agent(llm, tools)
    
    print("🤖 Agent is ready! Type 'quit' or 'exit' to end the chat.")
    print("-" * 50)
    
    while True:
        # Get user input from command line
        user_input = input("YOU: ")
        if user_input.lower() in ['quit', 'exit']:
            print("🤖 Goodbye!")
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"AGENT: {response['output']}")
        except Exception as e:
            # In case of an error, print it and continue the loop
            print(f"Sorry, an error occurred: {e}")
    print("-" * 50)

if __name__ == "__main__":
    main()
