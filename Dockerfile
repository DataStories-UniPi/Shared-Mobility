# Use the official Python 3.11 slim image
FROM python:3.11.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm requirements.txt  # Clean up

# Copy application code and related files
COPY . /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Use a dynamic entry point for flexibility
CMD ["python", "src/orchestrator.py"]

