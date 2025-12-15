FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

# System deps (optional but useful for some pandas cases)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY csv_labeling_tool_app.py /app/csv_labeling_tool_app.py

# Data directory for current.csv (persist via volume)
RUN mkdir -p /app/data

EXPOSE 5000

# Run
CMD ["python", "csv_labeling_tool_app.py"]
