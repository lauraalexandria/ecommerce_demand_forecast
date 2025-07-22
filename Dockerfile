FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
# COPY ./app ./app
COPY ./data ./data
COPY ./scr ./scr
COPY ./final_model ./final_model

RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python", "./app/scr/data_preparation.py"]
