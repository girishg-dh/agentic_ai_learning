from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from trip_planner_core import get_trip_plan

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class ChatMessage(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    if chat_message.message:
        response = get_trip_plan(chat_message.message)
        return JSONResponse(content={"response": response})
    return JSONResponse(content={"response": "No message provided"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)