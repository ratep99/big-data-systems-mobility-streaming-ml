docker container rm SparkTraining

docker build --rm -t bde/spark-training .

docker run --name SparkTraining --net bde -p 4040:4040 bde/spark-training 