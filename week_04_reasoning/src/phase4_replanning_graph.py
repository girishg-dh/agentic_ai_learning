# phase2_planner.py
import operator

#from os import getenv
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

#from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Load the .env file
load_dotenv()


# --- 1. Define the State ---
# state to gather all info throughout the processing 
class TripState(TypedDict):
    """The state of the trip planning agent"""
    messages: Annotated[list[BaseMessage], operator.add]
    replan_count: int
 

# --- 2. Define Tools and Nodes ---

# Initialize Search Tool
search_tool = TavilySearch(max_results=3)
tool_node = ToolNode([search_tool])

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# llm = ChatOpenAI(
#     base_url="http://localhost:1234/v1",
#     model="gpt-oss-20b",
#     api_key="lm-studio",
#     temperature=0
#     )


def create_specialist_node(system_prompt: str):
    """Helper to create a specialist node."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("placeholder", "{messages}")]
    )
    llm_with_tools = llm.bind_tools([search_tool])
    chain = prompt | llm_with_tools

    def node(state: TripState):
        return {"messages": [chain.invoke({"messages": state["messages"]})]}

    return node


agent_node = create_specialist_node(
    "You are a helpful travel planning assistant. Your goal is to find the best flights, hotels, "
    "and create an itinerary based on the user's request. Use the tools provided to find the "
    "necessary information. Once you have a final answer, respond directly to the user without "
    "calling any more tools."
)


replanner_node = create_specialist_node(
    (
        "You are an expert planner. The user has indicated that the previous attempt "
        "was unsuccessful. Analyze the conversation history and the last tool outputs. "
        "Formulate a new, single-sentence plan of action to achieve the user's goal. "
        "Your response should be a concise plan, not the answer itself."
    )
)


def human_in_loop_node(state: TripState):
    """Pauses the graph to ask for human feedback."""
    last_message = state['messages'][-1]
    if isinstance(last_message, ToolMessage):
        print("\n--- Tool Output ---")
        print(last_message.content)
    elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("\n--- AI Requesting Tool ---")
        print(last_message.tool_calls[0]['args'])
    else:
        print("\n--- AI Response ---")
        print(last_message.content)
    user_input = input("Is this correct? (y/n/r for yes/no/replan): ")
    if user_input.lower() == "y":
        return {"messages": [HumanMessage(content="User approved the step.")]}
    elif user_input.lower() == "n":
        return {
            "messages": [HumanMessage(content="User rejected the step. Halting")]
            }
    else:
        current_replan_count = state.get("replan_count", 0)
        return {
            "messages": [HumanMessage(content="User requested a replan.")],
            "replan_count": current_replan_count + 1
            }
    

MAX_REPLANS = 3


def replan_router(state: TripState):
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        if "rejected" in last_message.content:
            return END
        if "approved" in last_message.content:
            return "execute_tools"
        if state.get("replan_count", 0) >= MAX_REPLANS:
            return END
        if "replan" in last_message.content:
            return "replanner"
    # If the AI has tool calls, ask for feedback. If not, end.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "human_in_loop"
    return END


# --- 3. Define the Graph ---
workflow = StateGraph(TripState)

workflow.add_node("agent", agent_node)
workflow.add_node("replanner", replanner_node)
workflow.add_node("human_in_loop", human_in_loop_node)
workflow.add_node("execute_tools", tool_node)


workflow.set_entry_point("agent")
workflow.add_edge("replanner", "agent")
workflow.add_edge("execute_tools", "agent")

workflow.add_conditional_edges(
    "agent", 
    replan_router,
)
workflow.add_conditional_edges(
    "human_in_loop",
    replan_router,
)

# Compile the graph
app = workflow.compile()

# --- 4. Run the Graph ---
# --- 5. Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Trip Planner Agent ---")
    user_input = "Plan a 3-day trip to Berlin for two, focused on historical sites."
    initial_state = {"messages": [HumanMessage(content=user_input)], "replan_count": 0}

    final_state = None
    # Use invoke to run the graph and get the final state
    final_state = app.invoke(initial_state, {"recursion_limit": 50})

    print("\n--- FINAL AI RESPONSE ---")
    print(final_state['messages'][-1].content)
    print("\n--- Trip Planner Complete ---")