FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY ./app ./app
# COPY ./data ./data
# COPY ./scr ./scr
# COPY ./final_model ./final_model

RUN pip install fastapi uvicorn catboost pandas joblib

EXPOSE 8000

CMD ["uvicorn", "scr.api_csv:app", "--host", "0.0.0.0"]
