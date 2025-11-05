FROM python:3.11-slim

WORKDIR /app

RUN mkdir /app/cache && chmod 777 /app/cache

ENV TRANSFORMERS_CACHE="/app/cache"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8000", "--server.address=0.0.0.0"]