
import os

import autogen
import serpapi
from dotenv import load_dotenv

# --- Environment and Configuration ---
load_dotenv()

# Check for necessary API keys
serper_api_key = os.getenv("SERPER_API_KEY")

if not serper_api_key:
    raise ValueError("SERPER_API_KEY not found in environment variables.")

# AutoGen configuration for local Ollama
config_list = [
    {
        "model": "llama3.1:latest",
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": 42,  # Use a seed for caching
    "temperature": 0.7,
}


# --- Tool Definition ---
def search(query: str):
    """
    Performs a web search using the Serper API and returns the results.
    """
    try:
        return serpapi.search(q=query, api_key=serper_api_key)
    except Exception as e:
        return f"An error occurred during search: {str(e)}"


# --- Agent Definitions ---

# User Proxy Agent: Represents the user, executes code and tools.
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "education_guide", "use_docker": False},
    system_message="""A human user. You will execute the search tool when requested by other agents.
    Reply TERMINATE when the task is done.""",
)

# Researcher Agent: Finds information online.
researcher = autogen.AssistantAgent(
    name="Researcher",
    llm_config=llm_config,
    system_message="""You are a Senior Research Analyst. Your goal is to find the most relevant and up-to-date information.
    Your expertise lies in identifying top engineering universities in Germany (like the TU9 alliance), their specific admission requirements for Gymnasium students (NC, language certificates, required subjects), and application deadlines for the 2027 intake.
    Delegate the actual web search to the UserProxy by providing the search query.
    Analyze the search results and synthesize the key information. Do not present raw search results.
    """,
)

# Preparation Strategist Agent: Creates a plan based on research.
strategist = autogen.AssistantAgent(
    name="Preparation_Strategist",
    llm_config=llm_config,
    system_message="""You are an academic advisor specializing in German university admissions.
    Based on the researcher's findings, your goal is to create a detailed, step-by-step preparation timeline for a Berlin Gymnasium student aiming for 2027 admission.
    This plan must include:
    - Which Abitur subjects to focus on.
    - A timeline for taking language proficiency tests (like TestDaF or DSH).
    - A schedule for preparing and submitting application documents.
    - Tips for writing a strong personal statement (Motivationsschreiben).
    Ensure the plan is structured, actionable, and easy to follow.
    """,
)

# Writer Agent: Compiles the final report.
writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""You are a content creator who specializes in educational guides.
    Your task is to combine the list of universities from the researcher and the preparation plan from the strategist into a single, cohesive guide.
    The guide should be written in a clear, encouraging, and easy-to-read format, suitable for a high school student.
    Use clear headings, bullet points, and a friendly tone. The final output must be a complete markdown document.
    """,
)

# Critic Agent: Reviews the final output.
critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""You are an expert editor for educational content.
    Your goal is to review the generated guide for accuracy, clarity, and completeness.
    - Is the information correct and up-to-date?
    - Is the guide easy for a Gymnasium student to understand?
    - Is any crucial information missing (e.g., visa considerations for non-EU students, financial proof, accommodation)?
    Provide specific, actionable feedback for improvement. If the guide is satisfactory, state that it is ready for publication.
    """,
)

# --- Registering the tool ---
user_proxy.register_function(
    function_map={
        "search": search
    }
)

# --- Group Chat and Manager Setup ---
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, strategist, writer, critic],
    messages=[],
    max_round=15
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("## Welcome to the AutoGen Education Guide Crew")
    print("---------------------------------------------")

    # Ensure the working directory exists
    work_dir = "education_guide"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # The initial task prompt
    initial_prompt = (
        "Please create a comprehensive guide for a Berlin Gymnasium student who wants to apply for an "
        "engineering undergraduate program in Germany for the 2027 intake. "
        "The guide should cover the best universities, their admission requirements, and a detailed preparation plan. "
        "The final output should be a single, well-structured markdown document."
    )

    user_proxy.initiate_chat(
        manager,
        message=initial_prompt,
    )

    print("\n\n########################")
    print("## Here is the final result")
    print("########################\n")
    
    # The final result is usually the second-to-last message, as the last one is the termination signal.
    final_message = "No result found."
    if len(groupchat.messages) > 1:
        if groupchat.messages[-1]['content'].rstrip().endswith("TERMINATE"):
            final_message = groupchat.messages[-2]['content']
        else:
            final_message = groupchat.messages[-1]['content']

    print(final_message)

    # Save the final result to a file inside the specified work directory
    output_filename = os.path.join(work_dir, "education_guide_germany.md")
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(final_message)
    
    print(f"\nFinal guide saved to {output_filename}")


