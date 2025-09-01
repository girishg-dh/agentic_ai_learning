FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create directories for persistence
RUN mkdir -p /app/data /app/chat_history

# Expose port for web UI
EXPOSE 8000

# Run ingest first, then start the web server
CMD ["sh", "-c", "cd src && python ingest.py && python web_app.py"]