from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from airflow.models import Variable

default_args = {
    'owner':'kyungbin',
    'depends_on_past' :False,
    'start_date' : datetime(2024,1,1),
    'email_on_failure' : False,
    'email_on_retry' : False,
    'retries' : 1,
    'retry_delay' : timedelta(minutes=5)
}

dag = DAG(
    'model_training_seletion',
    default_args = default_args,
    description = 'a simple dag for model training and selection',
    schedule_interval = timedelta(days=1)
)

def feature_engineering(**kwargs):
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns= cancer.feature_names)
    y = pd.Series(cancer.target)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, stratify=y)

    # Xcom을 사용한 데이터 저장
    ti = kwargs['ti']
    ti.xcom_push(key = 'X_train', value = X_train.to_json())
    ti.xcom_push(key = 'X_test', value = X_train.to_json())
    ti.xcom_push(key = 'y_train', value = X_train.to_json(orient = 'records'))
    ti.xcom_push(key = 'y_test', value = X_train.to_json(orient = 'records'))

def train_model(model_name,**kwargs):
    ti = kwargs['ti']
    X_train = pd.read_json(ti.xcom_pull(key= 'X_train', task_ids = 'feature_engineering'))
    X_test = pd.read_json(ti.xcom_pull(key= 'X_test', task_ids = 'feature_engineering'))
    y_train = pd.read_json(ti.xcom_pull(key= 'y_train', task_ids = 'feature_engineering'), typ='series')
    y_test = pd.read_json(ti.xcom_pull(key= 'y_test', task_ids = 'feature_engineering'), typ='series')

    if model_name == 'RandomForestCLF':
        model = RandomForestClassifier(n_estimators=50, max_depth=8)
    elif model_name == 'GradinetboostingCLF':
        model = GradientBoostingClassifier(n_estimators=50, max_depth=8)
    else:
        raise ValueError("Unsupported model : "+ model_name)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    performance = accuracy_score(y_test, predictions)

    ti.xcom_push(key = f'performance_{model_name}', value = performance)


def select_best_model(**kwargs):
    ti = kwargs['ti']
    rf_performance = ti.xcom_pull(key='performance_RandomForestCLF', task_ids = 'train_rf')
    gb_performance = ti.xcom_pull(key='performance_GradinetboostingCLF', task_ids = 'train_gb')

    best_model = 'Radomforest' if rf_performance > gb_performance else 'GradientBoosting'
    print(f"Best model is {best_model} with performance {max(rf_performance, gb_performance)}")

    return best_model



with dag:
    t1 = PythonOperator(
        task_id = 'feature_engineering',
        python_callable = feature_engineering
    )
    t2 = PythonOperator(
        task_id = 'train_rf',
        python_callable = train_model,
        op_kwargs = {'model_name':'RandomForestCLF'},
        provide_context = True
    )
    t3 = PythonOperator(
        task_id = 'train_gb',
        python_callable = train_model,
        op_kwargs = {'model_name':'GradientboostingCLF'},
        provide_context = True
    )

    t4 = PythonOperator(
        task_id = 'select_best_model',
        python_callable=select_best_model,
        provide_context= True
    ) 

    t1 >> [t2,t3] >>t4