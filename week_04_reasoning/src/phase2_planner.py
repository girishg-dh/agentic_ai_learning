# phase2_planner.py
import operator
from os import getenv
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

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
# llm = ChatOpenAI(
#     api_key=getenv("OPENROUTER_API_KEY"),
#     base_url=getenv("OPENROUTER_BASE_URL"),
#     model="qwen/qwen3-coder:free",
#     default_headers={
#         "HTTP-Referer": "Planung", 
#         "X-Title": "Learner", 
#     },
#     temperature=0
# )

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="gpt-oss-20b",
    api_key="lm-studio",
    temperature=0
    )


# Helper to create specialist agent node
# In phase2_planner_corrected.py

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
        # --- ADDED DEBUGGING ---
        try:
            print(f"--- Calling Specialist: {system_prompt[:40]}... ---")
            result = chain.invoke({"messages": state["messages"]})
            return {"messages": [result]}
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
# --- 3. Define the Graph ---
workflow = StateGraph(TripState)

workflow.add_node("flight_researcher", flight_researcher)
workflow.add_node("hotel_researcher", hotel_researcher)
workflow.add_node("itinerary_creator", itinerary_creator)

workflow.set_entry_point("flight_researcher")
workflow.add_edge("flight_researcher", "hotel_researcher")
workflow.add_edge("hotel_researcher", "itinerary_creator")
workflow.add_edge("itinerary_creator", END)


# Compile the graph
app = workflow.compile()

# --- 4. Run the Graph ---
user_input = (
        "I want to plan a 5-day trip to Tokyo, "
        "Japan for two people in mid-October. "
        "My budget for flights and hotel is $3000."
)

initial_state = {"messages": [HumanMessage(content=user_input)]}

print("--- Starting Trip Planner ---")
for event in app.stream(initial_state, stream_mode="values"):
    if "flight_researcher" in event:
        print("\n--- Flight Research Complete ---")
        # The result from a tool-bound LLM is an AIMessage with tool_calls
        print(event["flight_researcher"]["messages"][-1].tool_calls)
    elif "hotel_researcher" in event:
        print("\n--- Hotel Research Complete ---")
        print(event["hotel_researcher"]["messages"][-1].tool_calls)
    elif "itinerary_creator" in event:
        print("\n--- Itinerary Creation Complete ---")
        print(event["itinerary_creator"]["messages"][-1].content)

print("\n--- Trip Planner Complete ---")
