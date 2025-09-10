# backend/tasks.py
from crewai import Task

from .agents import analyst, researcher

# Task 1: Research
research_task = Task(
  description=(
    "Gather and analyze the latest financial news, stock performance, and market sentiment "
    "for the company: {company}. Focus on the last quarter. "
    "Your final answer MUST be a thorough summary of the key findings."
  ),
  expected_output=(
    "A comprehensive report including: "
    "1. An overview of the company's recent stock performance. "
    "2. A summary of the latest significant news and events. "
    "3. An analysis of market sentiment."
    "4. Key financial metrics from their last quarterly report."
  ),
  agent=researcher
)

# Task 2: Analysis
analysis_task = Task(
  description=(
    "Analyze the research findings for {company} and provide a detailed investment recommendation. "
    "Consider the company's strengths, weaknesses, opportunities, and threats (SWOT analysis). "
    "Your analysis should be balanced, covering both potential risks and rewards."
  ),
  expected_output=(
    "A final investment analysis report for {company} that includes: "
    "1. A summary of the research findings. "
    "2. A SWOT analysis. "
    "3. A final investment recommendation (e.g., Buy, Hold, Sell). "
    "4. Justification for your recommendation, citing specific data points."
  ),
  agent=analyst
)
