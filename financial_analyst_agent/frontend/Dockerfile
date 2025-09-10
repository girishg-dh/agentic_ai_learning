# frontend/Dockerfile

# Use the same Python base image for consistency
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy and install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]