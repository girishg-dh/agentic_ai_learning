from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from main import run_research_crew

app = FastAPI(
    title="Research Assistant API",
    description="An API for the Research Assistant.",
    version="1.0.0"
)


templates = Jinja2Templates(directory="templates")



class ChatMessage(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def handle_chat(chat_message: ChatMessage):
    """Handles the chat message from the user and runs the crew."""
    print(f"Received brief: {chat_message.message}")
    report = run_research_crew(brief=chat_message.message)
    return {"response": report}