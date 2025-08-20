# Use Python 3.12
FROM python:3.12.18-slim

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run your app
CMD ["python", "main.py"]

