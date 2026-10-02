FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests

ENV PYTHONPATH=/app/src
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "a11_llm_optimizer.api:app", "--host", "0.0.0.0", "--port", "8080"]
