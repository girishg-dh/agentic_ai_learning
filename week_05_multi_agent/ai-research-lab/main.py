import os
import sys

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


def run_university_admissions_crew():
    llm = get_llm()
    search_tool = SerperDevTool()
    
    # --- Agent Definitions ---
    researcher = Agent(
        role='German CS/AI University Admissions Researcher',
        goal='Find the most up-to-date admission requirements and preparation recommendations for top Computer Science and AI-focused programs in Germany (especially TU9 universities) for the 2027 intake.',
        backstory=(
            "You are an expert in German higher education, specializing in engineering and computer science fields. "
            "You understand the TU9 alliance and other top universities offering CS/AI programs. "
            "You know the admissions landscape for Abitur students in Berlin and can identify additional preparation "
            "steps (like coding competitions, language skills, or math focus) that strengthen applications."
        ),
        llm=llm,
        verbose=True,
        tools=[search_tool],
    )

    analyst = Agent(
        role='CS/AI Admissions Data Analyst',
        goal='Extract, structure, and compare admissions information for Computer Science and AI-related undergraduate programs.',
        backstory=(
            "You are a detail-oriented analyst with expertise in higher education admissions. "
            "You specialize in distilling requirements for competitive CS/AI programs. "
            "You will highlight NC scores, prerequisite Abitur subjects (math, informatics, physics), "
            "language certificate requirements (German/English), application timelines, and any program-specific prerequisites."
        ),
        llm=llm,
        verbose=True
    )

    writer = Agent(
        role='University Admissions & Preparation Counselor',
        goal='Create a clear, motivating, and practical guide for a Berlin Gymnasium student aiming for CS/AI programs.',
        backstory=(
            "You are an approachable counselor who knows how to explain complex requirements in a student- and parent-friendly way. "
            "You combine admissions facts with preparation advice (study habits, extracurriculars, competitions, coding projects, language prep). "
            "Your reports reduce stress and give families confidence about the next steps."
        ),
        llm=llm,
        verbose=True
    )

    # --- Task Definitions ---
    research_task = Task(
        description=(
            "Search for admission requirements and preparation recommendations for undergraduate Computer Science "
            "and AI-focused engineering programs at the TU9 universities and other top German universities. "
            "Focus on the 2027 winter semester intake for students with a Berlin Abitur. "
            "Collect information on Numerus Clausus (NC), language requirements (German/English), required Abitur subjects, "
            "application deadlines, and any enrichment opportunities (special tracks, honors, international options). "
            "Also gather official links and sources."
        ),
        expected_output=(
            "A set of raw notes, links, and data about CS/AI admissions requirements and preparation recommendations "
            "from university websites and official portals."
        ),
        agent=researcher
    )

    analysis_task = Task(
        description=(
            "Analyze the research findings and structure them into a comparative format. "
            "For each university, extract: "
            "1. NC from the last available year for Computer Science and AI-related programs. "
            "2. German and/or English language requirements. "
            "3. Specific Abitur subjects expected (e.g., Math, Physics, Informatics). "
            "4. Application deadlines for 2027 (or most recent year, with note). "
            "5. Special preparation recommendations (coding contests, early internships, online courses)."
        ),
        expected_output=(
            "A structured markdown report with tables per university, summarizing NC, language, subject requirements, "
            "deadlines, and preparation tips."
        ),
        agent=analyst,
        context=[research_task]
    )

    writing_task = Task(
        description=(
            "Using the analyst’s report, create a comprehensive but student-friendly guide for a Berlin 12th grader. "
            "Start with an introduction to TU9 and other top universities for CS/AI. "
            "Provide a clear, university-by-university breakdown of requirements. "
            "Include preparation advice: which Abitur subjects to focus on, language test strategies, coding competitions, "
            "and extracurriculars that strengthen an AI/CS application. "
            "Conclude with an application timeline and motivational advice."
        ),
        expected_output=(
            "A polished, accessible admissions and preparation guide in markdown format. "
            "It should be titled 'Your Guide to Applying for Computer Science & AI Programs in Germany' "
            "and be ready to share with the student."
        ),
        agent=writer,
        context=[analysis_task]
    )


    # --- Crew Definition ---
    admissions_crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )
    
    result = admissions_crew.kickoff()
    return result




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
    return result.raw

# --- Main Execution Block ---
# if __name__ == "__main__":
#     print("## Welcome to the AI Research Lab Crew")
#     print("---------------------------------------")
#     default_topic = "AI in Healthcare"
#     final_result = run_ai_research_lab(default_topic)
    
#     print("\n\n########################")
#     print("## Here is the final result")
#     print("########################\n")
#     print(final_result)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("## Welcome to the University Admissions Assistant")
    print("-------------------------------------------------")
    final_report = run_university_admissions_crew()
    
    print("\n\n########################")
    print("## Here is your final report")
    print("########################\n")
    print(final_report)