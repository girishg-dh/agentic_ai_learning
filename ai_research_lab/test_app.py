from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import asyncio

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def handle_chat(
    topic: str = Form(...),
    scope: str = Form(...)
):
    # Simple test response
    async def stream_generator():
        result = f"# Test Report\n\n**Topic:** {topic}\n\n**Scope:** {scope}\n\nThis is a test response to verify the streaming functionality is working."
        yield result

    return StreamingResponse(stream_generator(), media_type="text/plain")