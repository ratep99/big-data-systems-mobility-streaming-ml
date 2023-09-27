docker container rm SparkStreaming

docker build --rm -t bde/spark-streaming .

docker run --name SparkStreaming --net bde -p 4040:4040 bde/spark-streaming