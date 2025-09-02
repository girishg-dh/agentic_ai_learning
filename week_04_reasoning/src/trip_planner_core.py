from langchain_core.messages import HumanMessage
from phase4_replanning_graph import TripState, app


def get_trip_plan(user_input: str) -> str:
    """Get trip plan response from the agent."""
    initial_state: TripState = {
        "messages": [HumanMessage(content=user_input)],
        "replan_count": 0,
    }
    
    # Use invoke to run the agent and get the final response
    final_state = app.invoke(initial_state, {"recursion_limit": 50})
    
    # Extract the final AI message
    final_message = final_state["messages"][-1]
    
    if hasattr(final_message, "content"):
        return final_message.content
    else:
        return "Sorry, I couldn't process your request."