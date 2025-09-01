import os
import json
from operator import itemgetter

import langchain
import qdrant_client
from helper import check_and_download_file, check_qdrant_status
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import messages_from_dict, messages_to_dict

# Global chat history
chat_history = []

os.environ["TOKENIZERS_PARALLELISM"] = "false"
langchain.debug = False

# Setup LLM and retriever
llm = ChatOpenAI(
    base_url="http://host.docker.internal:1234/v1",
    model="gpt-oss-20b",
    api_key="lm-studio"
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
client = qdrant_client.QdrantClient(url=qdrant_url, prefer_grpc=False)

qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="research_assistant",
    embedding=embeddings
)

retriever = MultiQueryRetriever.from_llm(retriever=qdrant_store.as_retriever(), llm=llm)

# Setup router
router_template = (
    "Given the conversation history and a follow up question, determine if the "
    "question is related to the context of the history.\n"
    "You must answer with only 'YES' or 'NO'.\n\n"
    "<history>\n"
    "{chat_history}\n"
    "</history>\n\n"
    "Question: {question}\n"
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["chat_history", "question"],
)
router_chain = router_prompt | llm | StrOutputParser()

# RAG chain
rag_template = (
    "Answer the user's question based on the context provided.\n"
    "If the context does not contain the answer, state that you don't have enough "
    "information.\n\n"
    "Context:\n"
    "{context}\n\n"
    "Question:\n"
    "{question}\n"
)
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Conversational chain
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])
conversational_chain = conversational_prompt | llm | StrOutputParser()

# Full chain with routing
routing_condition = RunnableLambda(lambda x: "YES" in x["topic"].upper())
branch = RunnableBranch((routing_condition, conversational_chain), rag_chain)
full_chain = RunnablePassthrough.assign(topic=router_chain) | branch

# Load chat history
try:
    with open("/app/chat_history/chat_history.json", "r") as f:
        retrieved_history = json.load(f)
        chat_history = messages_from_dict(retrieved_history)
except FileNotFoundError:
    pass

def get_assistant_response(user_input):
    global chat_history
    
    response = full_chain.invoke({
        "question": user_input,
        "chat_history": chat_history
    })
    
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    
    # Save chat history
    os.makedirs("/app/chat_history", exist_ok=True)
    serializable_history = messages_to_dict(chat_history)
    with open("/app/chat_history/chat_history.json", "w") as f:
        json.dump(serializable_history, f, indent=4)
    
    return response