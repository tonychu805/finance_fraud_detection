FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY implementation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY implementation/ .

# Create necessary directories
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV DEMO_MODE=true

# Run the application
CMD ["python", "run.py", "prod"] 