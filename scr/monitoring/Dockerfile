FROM python:3.9-slim

WORKDIR /app

COPY ../../data ./data
COPY ../../scr ./scr
COPY ../../mlflow.db ./mlflow.db
COPY ../../mlruns ./mlruns
COPY ../../Makefile.model Makefile.model


RUN pip install click mlflow pandas evidently prometheus_client time

CMD ["make", "-f", "Makefile.model", "monitor-test"]
