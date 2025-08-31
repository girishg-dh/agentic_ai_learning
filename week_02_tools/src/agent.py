# agent.py

from tools import get_current_weather, get_top_headlines, get_stock_price
from schemas import CityReport 

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

# 1. Define the LLM
#llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOpenAI(
    model="llama3.1", 
    temperature=0,
    base_url="http://localhost:11434/v1",  
    api_key="ollama" 
)

# 2. Instruct the LLM to use the CityReport schema
structured_llm = llm.with_structured_output(CityReport)

# 3. Define the prompt (remains the same)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. Create the tools list (remains the same)
tools = [get_current_weather, get_top_headlines, get_stock_price]

# 5. Create the agent using the structured LLM
agent = create_openai_tools_agent(structured_llm, tools, prompt)

# 6. Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. Invoke the agent with a multi-tool query
if __name__ == '__main__':
    print("--- Testing Multi-Tool Structured Output ---")
    response = agent_executor.invoke({
        "input": "Give me a full report on Berlin, including the weather and top news."
    })
    
    city_report = response['output']

    print("\n--- Final City Report (Structured) ---")
    print(f"Weather in {city_report['weather']['location']}:")
    print(f"  - Temp: {city_report['weather']['temperature']}°C")
    print(f"  - Desc: {city_report['weather']['description']}")
    
    print(f"\nTop News Headlines:")
    for article in city_report['news']:
        print(f"  - {article['title']}")