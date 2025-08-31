import os

import qdrant_client
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.retrievers import MultiQueryRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

os.environ["TOKENIZERS_PARALLELISM"] = "false"
langchain.debug = False

# --- 1. Setup the LLM ---
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="gpt-oss-20b",
    api_key="lm-studio"
    )
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = qdrant_client.QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=False
)

qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="research_assistant",
    embedding=embeddings
)

basic_retriver = qdrant_store.as_retriever()

multi_query_retriver = MultiQueryRetriever.from_llm(
    retriever=basic_retriver,
    llm=llm
)

# --- 3. Create the RetrievalQA Chain ---
# This chain does the following:
# 1. Takes your question.
# 2. Sends it to the retriever to fetch relevant document chunks.
# 3. Stuffs those chunks and your question into a prompt for the LLM.
# 4. Returns the LLM's answer.
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=multi_query_retriver
)


# --- 4. Start the Conversation ---
print("Your Personal Research Assistant is ready. "
      "Type 'exit' to end the chat.")
try:
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending the chat. Goodbye!")
            break
        # The 'invoke' method sends the input to the chain
        # and gets the response.
        response = rag_chain.invoke(input=user_input)
        print(f"Assistant: {response['result']}")
except KeyboardInterrupt:
    print("\nEnding the chat. Goodbye!")

