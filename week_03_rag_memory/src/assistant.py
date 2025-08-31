import os

import langchain
import qdrant_client
from helper import check_and_download_file, check_qdrant_status
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.retrievers import MultiQueryRetriever
#from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
chat_history = []

os.environ["TOKENIZERS_PARALLELISM"] = "false"
langchain.debug = False

# Make sure to execute ingest.py first
if not (check_qdrant_status() and check_and_download_file()):
    raise Exception("Failed to download the paper and/or Qdrant is not running.")

# --- 1. Setup the LLM and retriever ---
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

retriever = MultiQueryRetriever.from_llm(retriever=qdrant_store.as_retriever(), llm=llm)


# --- 2. Setup Memory ---
# We use ConversationBufferMemory to track the chat history.
# The 'input_key' and 'memory_key' are important for the router.
memory = ConversationBufferMemory(
    return_messages=True,
    input_key="question",
    memory_key="chat_history"
)

# For each route we will use
# template >> prompt >> chain
# --- 3. Define the Router ---
# The router decides whether to use the RAG chain or a conversational chain.
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
# The router chain itself
router_chain = (
    router_prompt
    | llm
    | StrOutputParser()
)


# --- 4. Define the RAG Chain (for answering from documents) ---
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

# The RAG chain
rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- 5. Define the Conversational Chain (for general chat) ---
conversational_template = (
    "You are a helpful AI assistant. Answer the user's question based on the "
    "conversation history.\n\n"
    "<history>\n"
    "{chat_history}\n"
    "</history>\n\n"
    "Question: {question}\n"
)
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])

conversational_chain = (
    conversational_prompt
    | llm
    | StrOutputParser()
)


# --- 6. Create the Full Chain with Router Logic ---
# The branch now directly pipes the input to the correct sub-chain.
# The `RunnablePassthrough` ensures the original question is passed to the chains.
routing_condition = RunnableLambda(lambda x: "YES" in x["topic"].upper())

branch = RunnableBranch(
    (routing_condition, conversational_chain),
    rag_chain
)

# We use `RunnablePassthrough.assign` to carry the original question along.
full_chain = (
    RunnablePassthrough.assign(
        topic=router_chain,
    )
    | branch
)

# --- 7. Start the Conversation ---
print("Your Personal Research Assistant is ready (Upgraded with Modern Chains).")

try:
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Invoke the full chain with the question and history
        response = full_chain.invoke({
            "question": user_input,
            "chat_history": chat_history
        })

        # Manually update our chat history list
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print(f"Assistant: {response}")

except KeyboardInterrupt:
    print("\nConversation ended.")