FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
# COPY ./app ./app
COPY ./data ./data
COPY ./project ./project
COPY ./final_model ./final_model

RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python", "./app/project/data_preparation.py"]
