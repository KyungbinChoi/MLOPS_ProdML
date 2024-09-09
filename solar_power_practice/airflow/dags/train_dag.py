from __future__ import annotations

import textwrap
from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonVirtualenvOperator
from tasks.train import train

with DAG(
    "train",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="train",
    schedule=timedelta(days=30),
    start_date=datetime(2021, 1, 1),
    catchup=False,
) as dag:
    train_task = PythonVirtualenvOperator(
        task_id="train_task",
        requirements=["tensorflow", "plotly", "scikit-learn", "numpy", "pandas", "kaleido"],
        python_callable=train
    )
    
    train_task