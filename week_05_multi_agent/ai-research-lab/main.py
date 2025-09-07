import sys
import os
from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# --- Environment and LLM Setup ---
load_dotenv()

def get_llm():
    # Your working get_llm function remains here
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if llm_provider == "ollama":
        print("Using Ollama model")
        from langchain_community.llms import Ollama
        return Ollama(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))
    elif llm_provider == "lmstudio":
        print("Using LM Studio model")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("LMSTUDIO_MODEL"), base_url=os.getenv("LMSTUDIO_BASE_URL"), api_key="lm-studio")
    else:
        print("Using Gemini model")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return LLM(model="gemini/gemini-2.0-flash-lite", temperature=0.7, api_key=gemini_api_key)

# --- Main Application Logic ---

def run_ai_research_lab(topic: str):
    """
    Runs the AI Research Lab crew for a given topic.
    """
    llm = get_llm()
    search_tool = SerperDevTool()

    # --- Agent Definitions ---
    researcher = Agent(role="Senior Research Analyst", goal=f'Uncover cutting-edge developments in {topic}', backstory='...', llm=llm, verbose=True, tools=[search_tool])
    writer = Agent(role='Tech Content Strategist', goal=f'Craft compelling content on {topic}', backstory='...', llm=llm, verbose=True)
    analyst = Agent(role='Principal Analyst', goal=f"Synthesize research findings about {topic} into a structured report.", backstory="...", llm=llm, verbose=True)
    critic = Agent(role='Expert Writing Critic', goal="Provide constructive feedback on the quality of the blog post about {topic}.", backstory="...", llm=llm, verbose=True)

    # --- Task Definitions ---
    research_task = Task(
        description=f"Identify the top 3 most significant trends in {topic} for 2026. Focus on advancements, multimodal models, and scientific discovery.",
        expected_output="A bulleted list of the top 3 trends, with a short paragraph for each.",
        agent=researcher
    )
    analyst_task = Task(
        description=f"Analyze the research findings on {topic}. For each trend, identify the core technology, its potential impact, and one key challenge.",
        expected_output="A detailed markdown report with an intro, three sections for each trend (tech, impact, challenge), and a summary.",
        agent=analyst,
        context=[research_task]
    )
    write_task = Task(
        description=f"Using the analyst's report on {topic}, write an engaging blog post for a non-technical audience. Make it witty and give it a catchy title.",
        expected_output="A 400-word blog post in markdown with a title.",
        agent=writer,
        context=[analyst_task]
    )
    critic_task = Task(
        description="Review the blog post. Check for clarity, accuracy, and engagement. Provide 3-5 actionable feedback points.",
        expected_output="A bulleted list of 3-5 constructive feedback points.",
        agent=critic,
        context=[write_task]
    )
    revision_task = Task(
        description="Revise the blog post based on the critic's feedback to make it publication-ready.",
        expected_output="The final, revised version of the blog post in markdown format.",
        agent=writer,
        context=[write_task, critic_task]
    )

    # --- Crew Definition ---
    static_crew = Crew(
        agents=[researcher, analyst, writer, critic],
        tasks=[research_task, analyst_task, write_task, critic_task, revision_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result = static_crew.kickoff()
    return result

# --- Main Execution Block ---
if __name__ == "__main__":
    print("## Welcome to the AI Research Lab Crew")
    print("---------------------------------------")
    default_topic = "AI in Healthcare"
    final_result = run_ai_research_lab(default_topic)
    
    print("\n\n########################")
    print("## Here is the final result")
    print("########################\n")
    print(final_result)