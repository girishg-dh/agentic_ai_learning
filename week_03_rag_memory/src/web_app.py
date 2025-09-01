import json
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict, messages_to_dict

# Import the assistant logic
from assistant_core import get_assistant_response, chat_history

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    response = get_assistant_response(message)
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)