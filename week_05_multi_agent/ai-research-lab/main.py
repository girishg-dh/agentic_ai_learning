import os
from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    # Your get_llm function remains here...
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


def run_research_crew(brief: str):
    llm = get_llm()
    search_tool = SerperDevTool()

    # --- Agent Definitions (Now more general) ---
    researcher = Agent(
        role='Senior Research Analyst',
        goal='To find the most relevant and up-to-date information based on a user\'s research brief.',
        backstory=(
            "You are a master of online research, skilled at uncovering hidden gems of information "
            "and synthesizing data from diverse sources to provide a comprehensive overview of any topic."
        ),
        llm=llm,
        verbose=True,
        tools=[search_tool]
    )
    analyst = Agent(
        role='Data Analyst',
        goal='To analyze research findings and structure them into a logical, easy-to-read format.',
        backstory=(
            "You are a meticulous analyst who excels at organizing complex information. You can identify "
            "key themes, data points, and narratives from raw research and present them as structured insights."
        ),
        llm=llm,
        verbose=True
    )
    writer = Agent(
        role='Lead Report Writer',
        goal='To compose a clear, concise, and engaging report from an analyst\'s structured findings.',
        backstory=(
            "You are a skilled writer, capable of transforming structured data and analysis into a well-written "
            "report that is both informative and easy for the target audience to understand."
        ),
        llm=llm,
        verbose=True
    )

    # --- Task Definitions (Now uses the full brief) ---
    research_task = Task(
        description=f"Conduct a thorough research based on the following brief and produce a detailed summary of your findings. \n\nBRIEF:\n{brief}",
        expected_output="A comprehensive summary of the research findings, including key facts, figures, and source URLs.",
        agent=researcher
    )
    analysis_task = Task(
        description="Analyze the provided research summary. Organize the information, identify the most critical insights, and structure it into a logical report format with clear headings.",
        expected_output="A structured report in markdown, with clear headings, bullet points, and a logical flow of information based on the research summary.",
        agent=analyst,
        context=[research_task]
    )
    writing_task = Task(
        description="Review the structured report and rewrite it into a final, polished research report. The tone should be professional and informative. Ensure it directly addresses all aspects of the original user brief.",
        expected_output="A final, well-written research report in markdown format that is ready for the user.",
        agent=writer,
        context=[analysis_task]
    )

    # --- Crew Definition ---
    research_crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result = research_crew.kickoff()
    return result.raw