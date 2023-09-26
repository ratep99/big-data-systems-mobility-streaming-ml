docker container rm StreamingApplication

docker build --rm -t bde/spark-streaming .

docker run --name StreamingApplication --net bde -p 4043:4043 bde/spark-streaming