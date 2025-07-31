FROM python:3.9-slim

WORKDIR /app

# COPY requirements.txt .
COPY ./data ./data
COPY ./scr ./scr
COPY ./mlflow.db ./mlflow.db
COPY ./mlruns ./mlruns

RUN pip install fastapi uvicorn pandas joblib mlflow

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "scr.api_csv:app", "--host", "0.0.0.0"]
