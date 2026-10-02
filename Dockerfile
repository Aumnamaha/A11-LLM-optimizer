FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests

ENV PYTHONPATH=/app/src
ENV HOST=0.0.0.0 PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn a11_llm_optimizer.api:app --host \"$HOST\" --port \"$PORT\""]
