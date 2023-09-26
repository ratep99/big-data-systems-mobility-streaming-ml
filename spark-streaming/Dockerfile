FROM bde2020/spark-python-template:3.1.2-hadoop3.2

RUN apk add --no-cache py3-numpy

RUN cd /app pip install -r requirements.txt
ENV SPARK_APPLICATION_PYTHON_LOCATION /app/app.py
ENV MODEL_LOCATION='hdfs://namenode:9000/dir/model'
ENV SCALER_LOCATION='hdfs://namenode:9000/dir/scaler'
ENV INDEXER_LOCATION='hdfs://namenode:9000/dir/indexer'
ENV KAFKA_URL=kafka:9092