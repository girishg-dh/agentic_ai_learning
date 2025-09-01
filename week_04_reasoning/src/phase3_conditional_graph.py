# phase2_planner.py
import operator
from os import getenv
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Load the .env file
load_dotenv()


# --- 1. Define the State ---
# state to gather all info throughout the processing 
class TripState(TypedDict):
    """The state of the trip planning agent"""
    user_request: Optional[str] = None
    flights: Optional[str] = None
    hotels: Optional[str] = None
    itinerary: Optional[str] = None
    messages: Annotated[list[BaseMessage], operator.add]
 

# --- 2. Define Tools and Nodes ---

# Initialize Search Tool
search_tool = TavilySearchResults(max_results=2)

# Initialize LLM
llm = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL"),
    model="z-ai/glm-4.5-air:free",
    default_headers={
        "HTTP-Referer": "Planung", 
        "X-Title": "Learner", 
    },
    temperature=0
)

# llm = ChatOpenAI(
#     base_url="http://localhost:1234/v1",
#     model="gpt-oss-20b",
#     api_key="lm-studio",
#     temperature=0
#     )




def create_specialist_node(system_prompt: str):
    """Create a node that acts as a specialist with a specific role."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    llm_with_tools = llm.bind_tools([search_tool])
    chain = prompt | llm_with_tools

    def node(state: TripState):
        try:
            print(f"--- Calling Specialist: {system_prompt[:40]}... ---")
            return {"messages": [chain.invoke({"messages": state["messages"]})]}

        except Exception as e:
            # This will catch and print any error during the LLM call
            print(f"!!! ERROR IN NODE: {e} !!!")
            # Return a message indicating failure
            error_message = HumanMessage(content=f"Error executing node: {e}")
            return {"messages": [error_message]}

    return node


# Create the specialist nodes using our helper function
flight_researcher = create_specialist_node(
    "You are a world-class flight research agent."
    " Find the best, most affordable flights "
    "for the user's request. Include airline, price, and "
    "flight times. Use the search tool."
)

hotel_researcher = create_specialist_node(
    "You are a world-class hotel research agent. "
    "Find the best hotels based on the user's "
    "request and flight details. Consider budget, "
    "location, and amenities. Use the search tool."
)

itinerary_creator = create_specialist_node(
    "You are a world-class travel planner. "
    "Create a detailed day-by-day itinerary "
    "based on the flight and hotel information provided."
    " Include suggested activities, "
    "restaurants, and travel tips."
)

# Helper to create specialist agent node
tool_node = ToolNode([search_tool])


# --- NEW: Define the Router Function ---
def should_continue(state: TripState):
    """
    Router function to decide the next step.

    Returns:
        "execute_tools": If the agent should use tools.
        "continue": If the agent is finished and should continue to the next step.
    """
    last_message = state["messages"][-1]
    # If the last message has tool calls, we need to execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    # Otherwise, we can continue to the next step
    return "continue"


# --- 3. Define the Graph ---
workflow = StateGraph(TripState)

workflow.add_node("flight_researcher", flight_researcher)
workflow.add_node("hotel_researcher", hotel_researcher)
workflow.add_node("itinerary_creator", itinerary_creator)
workflow.add_node("execute_tools", tool_node)

workflow.set_entry_point("flight_researcher")
workflow.add_conditional_edges(
    "flight_researcher", 
    should_continue,
    {"continue": "hotel_researcher", "execute_tools": "execute_tools"}
)
workflow.add_conditional_edges(
    "hotel_researcher", 
    should_continue,
    {"continue": "itinerary_creator", "execute_tools": "execute_tools"}
)
workflow.add_conditional_edges(
    "itinerary_creator", 
    should_continue,
    {"continue": END, "execute_tools": "execute_tools"}
)

workflow.add_edge("execute_tools", "flight_researcher")
workflow.add_edge("execute_tools", "hotel_researcher")
workflow.add_edge("execute_tools", "itinerary_creator")


# Compile the graph
app = workflow.compile()

# --- 4. Run the Graph ---
user_input = (
    "I want to plan a 7-day trip to Seoul, South Korea"
    " for one person in late October. "
    "My budget for flights and hotel is $2500."
)

initial_state = {"messages": [HumanMessage(content=user_input)]}

print("--- Starting Trip Planner ---")
for event in app.stream(initial_state, stream_mode="values"):
    # The `stream` method will now yield the state after each node,
    # including the tool execution steps.
    last_message = event["messages"][-1]
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            print(f"\n--- AI Requesting Tool: {last_message.tool_calls[0]['name']} ---")
            print(f"Query: {last_message.tool_calls[0]['args']}")
        else:
            print("\n--- AI Responding ---")
            print(last_message.content)
    elif isinstance(last_message, ToolMessage):
        print("\n--- Tool Responding ---")
        print(f"Tool: {last_message.name}\nOutput: {last_message.content[:200]}...")


print("\n--- Trip Planner Complete ---")
