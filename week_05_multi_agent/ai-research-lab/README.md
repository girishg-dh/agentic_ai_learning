# AI Research Lab

Multi-agent research assistant using CrewAI for generating comprehensive research reports.

## Features

- **Multi-Agent System**: Researcher, Analyst, and Writer agents collaborate
- **Web Interface**: Clean UI for submitting research briefs
- **Flexible LLM Support**: Gemini, Ollama, or LM Studio
- **Real-time Processing**: Live updates during report generation

## Quick Start

### 1. Environment Setup
```bash
cp .env.template .env
# Edit .env with your API keys
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
uvicorn api:app --reload
```
Open http://localhost:8000

## Configuration

### Environment Variables (.env)
```
# Required for web search
SERPER_API_KEY=your_serper_key

# LLM Provider (gemini/ollama/lmstudio)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key

# For Ollama
OLLAMA_MODEL=llama3.1:latest
OLLAMA_BASE_URL=http://localhost:11434

# For LM Studio
LMSTUDIO_MODEL=your_model
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

## Usage

1. Enter research topic
2. Define scope and key questions
3. Optional: Set timeline
4. Click "Generate Report"
5. Wait for multi-agent collaboration to complete

## Docker Deployment

```bash
docker build -t ai-research-lab .
docker run -p 8000:8000 --env-file .env ai-research-lab
```

## Kubernetes Deployment

```bash
minikube start
docker build -t ai-admissions-assistant .
kubectl apply -f deployment.yaml -f service.yaml
kubectl port-forward service/admissions-assistant-service 8080:8000
```

## Project Structure

- `main.py` - Core CrewAI logic and agent definitions
- `api.py` - FastAPI web server
- `templates/chat.html` - Web interface
- `autogen_main.py` - Alternative AutoGen implementation
- `main_*.py` - Development phases and variations

## Agent Roles

- **Researcher**: Finds relevant information using web search
- **Analyst**: Structures findings into logical reports
- **Writer**: Creates final polished research documents

