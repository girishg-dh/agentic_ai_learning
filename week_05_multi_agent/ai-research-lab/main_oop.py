
import os

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Import LLM providers conditionally
try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

class ResearchCrewManager:
    """
    Manages the setup and execution of a research crew using crewai.

    This class encapsulates the entire process of setting up the LLM,
    defining agents and tasks, and running the crew to generate a report.
    """

    def __init__(self, topic: str):
        """
        Initializes the ResearchCrewManager.

        Args:
            topic: The research topic for the crew to work on.
        """
        load_dotenv()
        self.topic = topic
        self.llm = self._setup_llm()
        self.search_tool = SerperDevTool()
        self._create_agents()
        self._create_tasks()

    def _setup_llm(self):
        """Selects and configures the LLM based on environment variables."""
        llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()

        if llm_provider == "ollama" and Ollama:
            print("Using Ollama model")
            return Ollama(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"))
        
        elif llm_provider == "lmstudio" and ChatOpenAI:
            print("Using LM Studio model")
            return ChatOpenAI(model=os.getenv("LMSTUDIO_MODEL"), base_url=os.getenv("LMSTUDIO_BASE_URL"), api_key="lm-studio")
            
        else:
            print("Using Gemini model")
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            # Assuming crewai has a way to handle Gemini directly or via a wrapper
            # For this example, we'll mock a generic LLM interface if crewai's internal one isn't exposed
            # In a real scenario, you would use the appropriate crewai LLM class
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key, temperature=0.7)

    def _create_agents(self):
        """Creates all the agents for the research crew."""
        self.researcher = Agent(
            role="Senior Research Analyst",
            goal=f'Uncover cutting-edge developments in {self.topic}',
            backstory=(
                "You are a world-class research analyst at a major tech think tank. "
                "Your expertise lies in identifying emerging trends and breaking news "
                "in the tech industry, particularly in AI."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool]
        )

        self.writer = Agent(
            role='Tech Content Strategist',
            goal='Craft compelling content on tech advancements',
            backstory=(
                "You are a renowned content strategist, known for distilling complex "
                "technical concepts into engaging, easy-to-understand blog posts. "
                "You transform raw data into narrative stories that captivate audiences."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

        self.analyst = Agent(
            role='Principal Analyst',
            goal=("Synthesize research findings into a structured, easy-to-digest report. "
                  "Identify key trends, connections, and actionable insights."),
            backstory=("You are a seasoned analyst at a top-tier consulting firm, renowned for "
                       "your ability to distill complex information into clear, concise, and "
                       "impactful reports. You excel at seeing the 'story' in the data."),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

        self.critic = Agent(
            role='Expert Writing Critic',
            goal="Provide constructive feedback on the quality of the blog post.",
            backstory=(
                "You are a seasoned editor at a prestigious tech journal. Your eye for detail is legendary. "
                "You are known for your ability to spot logical fallacies, unclear arguments, "
                "and opportunities to improve clarity and engagement. You provide feedback that is "
                "firm but fair, always aimed at elevating the quality of the work."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

        self.project_manager = Agent(
            role="Project Manager",
            goal=(
                "Efficiently manage the research project from conception to completion. "
                "Your primary role is to break down the main goal into smaller, "
                "manageable tasks and delegate them to the appropriate team members."
            ),
            backstory=(
                "You are a seasoned Project Manager with a knack for understanding complex "
                "requirements and orchestrating teams to achieve stellar results. You are a "
                "master of delegation, known for your ability to identify the right person "
                "for the right job and ensuring seamless collaboration."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

    def _create_tasks(self):
        """Creates all the tasks for the research process."""
        research_task = Task(
            description=(
                f"Identify the top 3 most significant trends in {self.topic}. "
                "Focus on advancements in multi-agent systems, multimodal models, and "
                "AI in scientific discovery. Your final answer must be a bulleted list "
                "of the trends, each with a brief explanation."
            ),
            expected_output=(
                "A bulleted list of the top 3 trends, with a short "
                "paragraph explaining each."
            ),
            agent=self.researcher
        )

        analyst_task = Task(
            description=(
                "Analyze the research findings provided on the top 3 trends. "
                "For each trend, identify the core technology, its potential impact, "
                "and one key challenge or risk. Your analysis should be structured "
                "and detailed, providing a solid foundation for a blog post."
            ),
            expected_output=(
                "A detailed report formatted in markdown. It should include an "
                "introduction, followed by three sections, one for each trend. "
                "Each section must contain the core technology, its potential impact, "
                "and a key challenge/risk. Conclude with a summary paragraph."
            ),
            agent=self.analyst,
            context=[research_task]
        )

        write_task = Task(
            description=(
                "Using the detailed analysis from the principal analyst, write a concise "
                "and engaging blog post about the top 3 trends. The post "
                "should be easy for a non-technical audience to understand. Make it "
                "witty and give it a catchy title."
            ),
            expected_output="A 400-word blog post with a title, formatted in markdown.",
            agent=self.writer,
            context=[analyst_task]
        )

        critic_task = Task(
            description=(
                "Review the blog post written by the Tech Content Strategist. "
                "Check for clarity, factual accuracy, engagement, and tone. "
                "Ensure the post is easily understandable for a non-technical audience. "
                "Provide a list of 3-5 specific, actionable feedback points for improvement."
            ),
            expected_output=(
                "A bulleted list of 3-5 constructive feedback points, "
                "explaining what could be improved and why."
            ),
            agent=self.critic,
            context=[write_task]
        )

        revision_task = Task(
            description=(
                "Revise the blog post based on the constructive feedback provided by the Expert Writing Critic. "
                "Incorporate the suggestions to improve clarity, engagement, and accuracy. "
                "The final version should be polished and ready for publication."
            ),
            expected_output=(
                "The final, revised version of the blog post in a 400-word markdown format, "
                "incorporating the critic's feedback."
            ),
            agent=self.writer,
            context=[write_task, critic_task]
        )

        # The main task for the project manager to oversee.
        self.manager_task = Task(
            description=(
                f"Oversee the creation of a comprehensive blog post on {self.topic}. "
                "The final output must be a polished, well-researched, and engaging article. "
                "Your job is to break down this goal and delegate the work to your team of specialists."
            ),
            expected_output=(
                "The final, revised version of the blog post in a 400-word markdown format."
            ),
            agent=self.project_manager
        )

    def run(self):
        """Assembles and runs the crew, returning the final result."""
        managerial_crew = Crew(
            agents=[self.project_manager, self.researcher, self.analyst, self.writer, self.critic],
            tasks=[self.manager_task],
            process=Process.sequential,
            verbose=True,
        )
        result = managerial_crew.kickoff()
        return result

# --- Main Execution Block ---
if __name__ == "__main__":
    print("## Welcome to the OOP AI Research Lab Crew")
    print("---------------------------------------")

    # The research topic can be easily changed here.
    research_topic = "AI in 2026, focusing on multi-agent systems, multimodal models, and AI in scientific discovery"
    
    # Instantiate and run the crew manager.
    crew_manager = ResearchCrewManager(topic=research_topic)
    final_result = crew_manager.run()

    print("\n\n########################")
    print("## Here is the final result")
    print("########################\n")
    print(final_result)

    # Optionally, save the result to a file.
    with open("final_report_oop.md", "w") as f:
        f.write(final_result)
