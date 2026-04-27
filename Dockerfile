# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure data and models directories exist for the pipeline
RUN mkdir -p data/raw data/processed models reports/figures

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
# We use uvicorn to serve the FastAPI app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
