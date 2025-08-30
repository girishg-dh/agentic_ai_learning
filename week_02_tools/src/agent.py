from tools import get_current_weather, get_top_headlines, get_stock_price

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

# 1. Define the LLMs
#llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm = ChatOpenAI(temperature=0, model="openai/gpt-oss-20b", base_url="http://localhost:1234/v1", max_retries=3)

# 2. Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 3. Create a list of tools
tools = [get_current_weather, get_top_headlines, get_stock_price]

# 4. Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# 5. Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Invoke the agent executor with a query
if __name__ == '__main__':
    print("Agent is ready! Ask a question about the weather.")

    print("--- Testing Stock Price Tool ---")
    response = agent_executor.invoke({"input": "What is the stock price for Apple?"})
    print("\nFinal Answer:")
    print(response["output"])
    
    print("\n" + "="*50 + "\n")

    response = agent_executor.invoke({"input": "What's the weather like in Berlin?"})
    print("\nFinal Answer:")
    print(response["output"])
    print("--- Testing Weather Tool ---")
    response = agent_executor.invoke({"input": "What's the weather like in Tokyo?"})
    print("\nFinal Answer:")
    print(response["output"])

    print("\n" + "="*50 + "\n")

    print("--- Testing News Tool ---")
    response = agent_executor.invoke({"input": "What are the top headlines in India?"})
    print("\nFinal Answer:")
    print(response["output"])