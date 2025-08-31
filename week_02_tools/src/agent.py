"""
Agent module that creates a structured output agent using LangChain
and a local gpt-oss model via prompt engineering.
"""
import json

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import CityReport  # We still use this for validation/parsing
from tools import get_current_weather, get_stock_price, get_top_headlines

# --- NEW: A detailed prompt that describes the desired JSON output ---
SYSTEM_PROMPT = """You are a helpful assistant. You must use the provided tools to gather the information required to answer the user's questions.

After you have used the tools and have all the necessary information, you MUST format your final answer as a single, valid JSON object. This JSON object must strictly adhere to the following structure:

{{
  "weather": {{
    "location": "The city and country name e.g. 'Berlin, Germany'",
    "temperature": "The current temperature in Celsius as a float",
    "description": "A brief description of the weather conditions"
  }},
  "news": [
    {{
      "title": "The title of a news article",
      "url": "The URL to the full news article"
    }}
  ],
  "error_message": "An optional field for any error messages. Null if no errors."
}}

For all other questions (e.g., "what's the weather in London?", "what's the price of NVDA?"), answer directly and concisely in plain text based on the output from the tools. Do not use the JSON format for these simple questions.

Do not include any other text, explanations, or markdown formatting around the final JSON object.
"""

def setup_agent():
    """
    Sets up and configures the agent for the gpt-oss model.
    Returns an AgentExecutor instance.
    """
    llm = ChatOpenAI(
        model="gpt-oss-20b",
        temperature=0,
        base_url="http://localhost:1234/v1",
        api_key="local"
    )
    
    # We now use our new detailed prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [get_current_weather, get_top_headlines, get_stock_price]

    # We use the standard llm, NOT a structured_llm
    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def print_city_report(response_str: str):
    """
    Parses the JSON string from the agent and prints it.
    
    Args:
        response_str (str): The agent's raw string output, expected to be JSON.
    """
    print("\n--- Raw Agent Output ---")
    print(response_str)

    try:
        # The output is now a string that we need to parse into a dictionary
        report = json.loads(response_str)
        
        if report.get("error_message"):
            print(f"\n--- Error Reported by Agent ---")
            print(report["error_message"])
            return

        print("\n--- Parsed City Report ---")
        if "weather" in report and report["weather"]:
            weather = report["weather"]
            print(f"Weather in {weather.get('location', 'N/A')}:")
            print(f"  - Temp: {weather.get('temperature', 'N/A')}°C")
            print(f"  - Desc: {weather.get('description', 'N/A')}")
        
        if "news" in report and report["news"]:
            print(f"\nTop News Headlines:")
            for article in report["news"]:
                print(f"  - {article.get('title', 'No Title')}")

    except json.JSONDecodeError:
        print("\n--- Error ---")
        print("Failed to decode the agent's output. The response was not valid JSON.")
    except Exception as e:
        print(f"\nAn error occurred while printing the report: {e}")


if __name__ == '__main__':
    # --- The Final CLI Application Loop ---
    
    print("✅ Your Multi-Tool Assistant is ready! Type 'exit' or 'quit' to end.")
    
    # Setup the agent once
    agent_executor = setup_agent()
    
    while True:
        # Get user input from the command line
        user_input = input("\n> ")
        # Check for exit commands
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting assistant. Goodbye! 👋")
            break
        # Invoke the agent with the user's input
        try:
            # Turn off verbose for a cleaner chat experience
            agent_executor.verbose = False 
            response = agent_executor.invoke({
                "input": user_input
            })
            # Print the structured report
            print_city_report(response['output'])
        except Exception as e:
            print(f"\nAn error occurred: {e}")