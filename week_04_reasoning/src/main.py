# main.py
from langchain_core.messages import HumanMessage
from phase4_replanning_graph import TripState, app


def main():
    """The main entry point for the trip planner application."""
    print("🤖 Welcome to the Intelligent Trip Planner!")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("-" * 50)

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("🤖 Goodbye!")
            break

        if user_input.strip() == "":
            continue

        # Define the initial state for the graph
        initial_state: TripState = {
            "messages": [HumanMessage(content=user_input)],
            "replan_count": 0,
        }

        print("🤖 Planning your trip, please wait...")
        
        # Use invoke to run the agent and get the final response
        final_state = app.invoke(initial_state, {"recursion_limit": 50})

        # Extract and print the final AI message
        final_message = final_state["messages"][-1]
        
        # The final message could be a tool call if something went wrong, 
        # or the AI's content response.
        if hasattr(final_message, "content"):
            print(f"\nPlanner:\n{final_message.content}")
        
        print("-" * 50)

# Run the main application loop
if __name__ == "__main__":
    main()