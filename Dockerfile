# In Dockerfile

FROM python:3.11-slim

WORKDIR /app

# --- ADD THIS LINE ---
# Create a writable cache directory before setting the ENV
RUN mkdir /app/cache && chmod 777 /app/cache

# Set a writable cache directory for Hugging Face models
ENV TRANSFORMERS_CACHE="/app/cache"

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8000

# Launch the Streamlit UI as the main application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8000", "--server.address=0.0.0.0"]