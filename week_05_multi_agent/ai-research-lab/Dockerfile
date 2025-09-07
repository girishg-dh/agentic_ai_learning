FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install uv, then use uv pip to install dependencies system-wide
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

# Small fix: missing comma between host and port args
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
