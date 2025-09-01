import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph


# --- 1. Define the State ---
# The state is the shared memory of our graph.
# A state that has a single key, "messages".
# The `Annotated` part with `operator.add` tells LangGraph that
# whenever a node returns a "messages" key, it should be ADDED
# to the existing list, not overwrite it. This is how we build conversation 
# history.
class AgentState(TypedDict):
    """The state of the agent"""
    messages: Annotated[list[BaseMessage], operator.add]


# --- 2. Define the Nodes ---
# Nodes are the "workers" of the graph. They are just Python functions
# that take the state as input and return a dictionary with updates to the state.
def node_a(state: AgentState):
    """A node that performs first step"""
    print("----EXECUTING A NODE----")
    last_message = state["messages"][-1]
    return {"messages": [f"Node A processed: '{last_message}"]}

def node_b(state: AgentState):
    """A node that performs second step"""
    print("----EXECUTING B NODE----")
    last_message = state["messages"][-1]
    return {"messages": [f"Node B processed: '{last_message}"]}

# --- 3. Build the Graph ---
# Initialize a new StateGraph with our AgentState
workflow = StateGraph(AgentState)

# Add the nodes to the graph. The first argument is a unique name for the node,
# and the second is the function that implements the node.
workflow.add_node("A", node_a)
workflow.add_node("B", node_b)


# Set the entry point of the graph. This is the first node to be executed.
workflow.set_entry_point("A")

# Add the edges. This defines the flow.
# We're creating a simple, unconditional edge from node A to node B.
workflow.add_edge("A", "B")

# Another edge from B to END
workflow.add_edge("B", END)

# --- 4. Compile and Run the Graph ---
# Compile the graph into a runnable object.
app = workflow.compile()

# Invoke initial state
initial_input = {"messages": ["Starting the processs..."]}
final_state = app.invoke(initial_input)

print("\n---FINAL STATE---")
print(final_state["messages"])