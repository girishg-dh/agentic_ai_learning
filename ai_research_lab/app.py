import os
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from main import run_research_crew
import tempfile

app = FastAPI(
    title="AI Research Assistant API",
    description="An API for the AI Research Assistant.",
    version="2.0.0"
)

templates = Jinja2Templates(directory="templates")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def handle_chat(
    topic: str = Form(...),
    scope: str = Form(...),
    timeline: str = Form(None),
    file: UploadFile = File(None)
):
    brief = f"**Topic:** {topic}\n**Scope:** {scope}"
    if timeline:
        brief += f"\n**Timeline:** {timeline}"

    file_path = None
    if file:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            file_path = tmp.name
        print(f"File saved to: {file_path}")

    report = run_research_crew(brief=brief, file_path=file_path)
    
    # Clean up the temp file after the crew has run
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    return {"response": report}