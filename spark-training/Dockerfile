FROM bde2020/spark-python-template:3.1.2-hadoop3.2

RUN apk add --no-cache py3-numpy

ENV SPARK_MASTER spark://spark-master:7077
ENV SPARK_APPLICATION_PYTHON_LOCATION /app/app.py
ENV SPARK_APPLICATION_MAIN_CLASS App3
ENV HDFS_BUS='hdfs://namenode:9000/dir/denverVehiclesCleaned.csv'
ENV MODEL_LOCATION='hdfs://namenode:9000/dir/model'
ENV SCALER_LOCATION='hdfs://namenode:9000/dir/scaler'
ENV INDEXER_LOCATION='hdfs://namenode:9000/dir/indexer'


ADD start.sh /

RUN chmod +x /start.sh
CMD ["/bin/bash", "/start.sh"]