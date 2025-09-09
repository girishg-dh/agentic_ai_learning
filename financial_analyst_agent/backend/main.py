from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Financial Analyst Agent API",
    description="An API for interacting with a CrewAI financial analyst agent.",
    version="0.1.0",
)

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint to confirm the server is running.
    """
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Financial Analyst Agent API!"}
