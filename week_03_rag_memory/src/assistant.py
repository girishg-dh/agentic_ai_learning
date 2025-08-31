from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI



# --- 1. Setup the LLM ---
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="gpt-oss-20b",
    api_key="lm-studio"
    )
# --- 2. Setup the Memory ---
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# --- 3. Create the Conversation Chain ---
conversation = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True
)

# --- 4. Start the Conversation ---
print("Your Personal Research Assistant is ready. Type 'exit' to end the chat.")
try:
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending the chat. Goodbye!")
            break
        # The 'invoke' method sends the input to the chain and gets the response.
        response = conversation.invoke(input=user_input)
        print(f"Assistant: {response['response']}")
except KeyboardInterrupt:
    print("\nEnding the chat. Goodbye!")