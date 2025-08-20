# Use Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port (adjust if needed)
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run your app
CMD ["python", "main.py"]

