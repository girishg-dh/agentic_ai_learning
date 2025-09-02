# Week 4: Reasoning - Trip Planner

An intelligent trip planning agent with web interface using LangGraph for complex reasoning workflows.

## Features

- Multi-agent reasoning with LangGraph
- Web-based chat interface
- Tool integration for real-time search
- Docker containerization
- Replanning capabilities

## Quick Start

### Prerequisites
- Docker and Docker Compose
- API keys for Google Gemini and Tavily (set in .env file)

### Run with Docker

1. **Set up environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

2. **Build and run:**
   ```bash
   cd week_04_reasoning
   docker-compose up --build
   ```

3. **Access the web interface:**
   - Trip Planner UI: http://localhost:8001

### Run Locally

1. **Install dependencies:**
   ```bash
   cd src
   pip install -r requirements.txt
   ```

2. **Run the web app:**
   ```bash
   python web_app.py
   ```

3. **Or run CLI version:**
   ```bash
   python main.py
   ```

## Project Structure

```
week_04_reasoning/
├── docker-compose.yml
├── README.md
└── src/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py                    # CLI interface
    ├── web_app.py                # FastAPI web interface
    ├── trip_planner_core.py      # Core planning logic
    ├── phase4_replanning_graph.py # Main agent implementation
    ├── phase1_simple_graph.py    # Learning examples
    ├── phase2_planner.py         # Learning examples
    ├── phase3_conditional_graph.py # Learning examples
    └── templates/
        └── chat.html             # Web UI template
```

## How It Works

The trip planner uses LangGraph to create a sophisticated reasoning workflow:

1. **Agent Node:** Processes user requests and calls tools
2. **Tool Integration:** Uses Tavily search for real-time information
3. **Replanning:** Can reformulate plans based on feedback
4. **Human-in-Loop:** Interactive feedback system (CLI only)

---

## Author

- [girishg-dh](https://github.com/girishg-dh)