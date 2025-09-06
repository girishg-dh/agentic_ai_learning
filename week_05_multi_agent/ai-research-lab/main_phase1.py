import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# --- Environment and LLM Setup ---

# Load environment variables from .env file
load_dotenv()


def get_llm():
    """Function to dynamically select the LLM based on environment variables."""
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
       
        return LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature=0.7,
            api_key=gemini_api_key
        )


# Select the LLM
llm = get_llm()

# --- Tool Setup ---
search_tool = SerperDevTool()

# --- Agent Definitions ---

researcher = Agent(
    role="Senior Research Analyst",
    goal='Uncover cutting-edge developments in AI and data science',
    backstory=(
        "You are a world-class research analyst at a major tech think tank. "
        "Your expertise lies in identifying emerging trends and breaking news "
        "in the tech industry, particularly in AI."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory=(
        "You are a renowned content strategist, known for distilling complex "
        "technical concepts into engaging, easy-to-understand blog posts. "
        "You transform raw data into narrative stories that captivate audiences."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --- Task Definitions ---

research_task = Task(
    description=(
        "Identify the top 3 most significant trends in AI in 2026. "
        "Focus on advancements in multi-agent systems, multimodal models, and "
        "AI in scientific discovery. Your final answer must be a bulleted list "
        "of the trends, each with a brief explanation."
    ),
    expected_output=(
        "A bulleted list of the top 3 AI trends for 2026, with a short "
        "paragraph explaining each."
    ),
    agent=researcher
)

write_task = Task(
    description=(
        "Using the research findings from the research analyst, write a concise "
        "and engaging blog post about the top 3 AI trends in 2026. The post "
        "should be easy for a non-technical audience to understand. Make it "
        "witty and give it a catchy title."
    ),
    expected_output="A 400-word blog post with a title, formatted in markdown.",
    agent=writer
)

# --- Crew Definition ---


def run_crew():
    """Creates and runs the research crew."""
    research_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result = research_crew.kickoff()
    return result

# --- Main Execution Block ---


if __name__ == "__main__":
    print("## Welcome to the AI Research Lab Crew")
    print("---------------------------------------")
    
    final_result = run_crew()
    
    print("\n\n########################")
    print("## Here is the final result")
    print("########################\n")
    print(final_result) 