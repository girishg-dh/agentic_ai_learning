# backend/main.py

import os

from crewai import Crew, Process
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agents import analyst, researcher
from .tasks import analysis_task, research_task

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


app = FastAPI(
    title="Financial Analyst Agent API",
    description=(
        "An API for interacting with a CrewAI financial analyst agent."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydentic Models ---
class AnalysisRequest(BaseModel):
    company: str


@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint to confirm the server is running.
    """
    return {"status": "ok"}


@app.post("/analyze", tags=["Financial Analysis"])
async def analyze_company(request: AnalysisRequest):
    """
    Endpoint to trigger the financial analysis of a given company.
    """

    try:
        financial_crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            process=Process.sequential,
            verbose=True,
        )
        inputs = {"company": request.company}
        result = financial_crew.kickoff(inputs=inputs)
        return {"result": str(result)}

    except Exception as e:
        return {"error": str(e)}
