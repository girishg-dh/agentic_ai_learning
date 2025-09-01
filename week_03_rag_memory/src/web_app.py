import json
import os

# Import the assistant logic
from assistant_core import (
    chat_history, 
    get_assistant_response, 
    clear_chat_history, 
    prune_chat_history
)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Pydantic model for the incoming message
class ChatMessage(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    # The chat_history is now managed per session, so we pass the current state
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    if chat_message.message:
        response = get_assistant_response(chat_message.message)
        # The last message in chat_history is the assistant's response
        assistant_message = chat_history[-1].content
        return JSONResponse(content={"response": assistant_message})
    return JSONResponse(content={"response": "No message provided"}, status_code=400)

@app.post("/chat/clear")
async def clear_history():
    clear_chat_history()
    return JSONResponse(content={"message": "Chat history cleared"})

@app.post("/chat/prune")
async def prune_history():
    prune_chat_history() # a default of 4 messages will be kept
    return JSONResponse(content={"message": "Chat history pruned"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)