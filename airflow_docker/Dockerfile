FROM python:3.10-slim

ENV AIRFLOW_HOME=/usr/local/airflow

RUN apt-get update && \
    apt-get install -y gcc libc-dev vim && \
    rm -rf /var/lib/apt/lists/*


RUN pip install apache-airflow

RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME
RUN airflow db init

COPY my_dag.py $AIRFLOW_HOME/dags/

EXPOSE 8080

CMD airflow webserver -p 8080 & airflow scheduler

