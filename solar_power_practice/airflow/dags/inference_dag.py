from __future__ import annotations

import textwrap
from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonVirtualenvOperator,ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from tasks.inference import inference
from tasks.is_model_drift import is_model_drift

with DAG(
    "inference",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="inference",
    schedule=timedelta(days=7),
    start_date=datetime(2021, 1, 1),
    catchup=False,
) as dag:
    inference_task = PythonVirtualenvOperator(
        task_id="inference_task",
        requirements=["tensorflow", "plotly", "scikit-learn", "numpy", "pandas", "kaleido"],
        python_callable=inference
    )
    
    is_model_drift_task = ShortCircuitOperator(
        task_id = "is_model_drift_task",
        python_callable = is_model_drift

    )

    train_trigger_task = TriggerDagRunOperator(
        task_id = "train_trigger_task",
        trigger_dag_id = "train"
    )

    inference_task >> is_model_drift_task >> train_trigger_task