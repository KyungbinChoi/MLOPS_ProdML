# 기존 airflow 이미지에서 확장
FROM apache/airflow:2.10.1

# 필요한 시스템 패키지 설치 (gcc, libhdf5-dev, pkg-config)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       libhdf5-dev \
       pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# airflow user로 변경
USER airflow
