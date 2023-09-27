from pyspark import SparkContext
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import StringIndexerModel, VectorAssembler, MinMaxScalerModel, StandardScalerModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.classification import NaiveBayesModel
from pyspark.sql.functions import concat, date_format, lit, to_timestamp, hour, when, col, from_json
from pyspark.sql.types import *

import os

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

      
dbhost = os.getenv('INFLUXDB_HOST')
dbport = os.getenv('INFLUXDB_PORT')
dbuser = os.getenv('INFLUXDB_USERNAME')
dbpassword = os.getenv('INFLUXDB_PASSWORD')
dbname = os.getenv('INFLUXDB_DATABASE')
MODEL_LOCATION = os.getenv('MODEL_LOCATION')
SCALER_LOCATION = os.getenv('SCALER_LOCATION')
INDEXER_LOCATION = os.getenv('INDEXER_LOCATION')
KAFKA_URL = os.getenv('KAFKA_URL')
KAFKA_TOPIC = 'vehicles_topic'


class InfluxDBWriter:
    print("Počinje štampanje!")
    def __init__(self):
        self._org = 'denver_vehicles'
        self._token = 'P9MQUbxzG7C8WwCfbCH6cKPXhLCcbOgUaWqSPth7qfDIjbZqmUOQPHtxDa1Qn2hio0v5orjl-Px4D0znn-gRvw=='
        self.client = InfluxDBClient(
            url = "http://influxdb:8086", token=self._token, org = self._org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def open(self, partition_id, epoch_id):
        print("Opened %d, %d" % (partition_id, epoch_id))
        return True

    def process(self, row):
        self.write_api.write(bucket='denver_vehicles',
                             record=self._row_to_line_protocol(row))

    def close(self, error):
        self.write_api.__del__()
        self.client.__del__()
        print("Closed with error: %s" % str(error))

    def _row_to_line_protocol(self, row):
        print(row)
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        return Point.measurement(KAFKA_TOPIC).tag("measure", KAFKA_TOPIC) \
                    .field("timestamp", float(row['timestamp'])) \
                    .field("id", float(row['id'])) \
                    .field("type", float(row['type'])) \
                    .field("latitude", float(row['latitude'])) \
                    .field("longitude", float(row['longitude'])) \
                    .field("speed_kmh", float(row['speed_kmh'])) \
                    .field("acceleration", float(row['acceleration'])) \
                    .field("distance", float(row['distance'])) \
                    .field("odometer", float(row['odometer'])) \
                    .field("pos", float(row['pos'])) \
                    .field("prediction", float(row['prediction'])) \
                    .time(timestamp, write_precision='ms')
    
def create_spark_session(app_name):
    """
    Create a Spark session with the given app name and set log level to ERROR.
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_models():
    """
    Load the machine learning models and scaler.
    """
    model = LinearRegressionModel.load(MODEL_LOCATION)
    scaler = StandardScalerModel.load(SCALER_LOCATION)
    indexer = StringIndexerModel.load(INDEXER_LOCATION)
    return model, scaler, indexer


if __name__ == '__main__': 
    
    appName = "Projekat3Stream"
    print("CREATING SESSION...")
    # Create a Spark session
    spark = create_spark_session(appName)
    print("LOAD MODELS AND SCALERS...")
    # Load models and scaler
    model, scaler, indexer = load_models()
    print("PARSING KAFKA DATA")
    # Define schema for parsing Kafka data
    schema = StructType([
        StructField("timestamp", StringType()),
        StructField("id", StringType()),
        StructField("type", StringType()),
        StructField("latitude", StringType()),
        StructField("longitude", StringType()),
        StructField("speed_kmh", StringType()),
        StructField("acceleration", StringType()),
        StructField("distance", StringType()),
        StructField("odometer", StringType()),
        StructField("pos", StringType())
    ])

    # Read data from Kafka
    df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_URL)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )
    print("TRANSFORMING DATA")
    # Transform the data
    parsed_values = df.select(
        "timestamp", from_json(col("value").cast("string"), schema).alias("parsed_values")
    )
    parsed_values.printSchema()
    df_org = parsed_values.select(
        "timestamp",
        "parsed_values.id",
        "parsed_values.type",
        "parsed_values.latitude",
        "parsed_values.longitude",
        "parsed_values.speed_kmh",
        "parsed_values.acceleration",
        "parsed_values.distance",
        "parsed_values.odometer",
        "parsed_values.pos"
    )
    df_org.printSchema()
    print(" DATA TRANSFORMED ")
    df_org = df_org.withColumn("latitude", col("latitude").cast("double"))
    df_org = df_org.withColumn("longitude", col("longitude").cast("double"))
    df_org = df_org.withColumn("speed_kmh", col("speed_kmh").cast("double"))
    df_org = df_org.withColumn("acceleration", col("acceleration").cast("double"))
    df_org = df_org.withColumn("distance", col("distance").cast("double"))
    df_org = df_org.withColumn("odometer", col("odometer").cast("double"))
    df_org = df_org.withColumn("pos", col("pos").cast("double"))

    # Manipulacija sa kolonom datetime
    df_org = df_org.withColumn("datetime", concat(col("timestamp"), lit("Z")))
    df_org = df_org.withColumn("datetime", to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss.SSSX"))
    df_org = df_org.drop("timestamp")

    # Formatiranje datuma
    df_org = df_org.withColumn("datetime", date_format("datetime", "yyyy-MM-dd HH:mm:ss"))

    # Prikazivanje rezultata
    #df_org.show()

    print("Poslednji df")
    df_org.printSchema()
    indexed = indexer.transform(df_org)

    columns = ["latitude", "longitude", "speed_kmh", "acceleration", "distance"]

    va = VectorAssembler().setInputCols(columns).setOutputCol('features').setHandleInvalid("skip").transform(indexed)

    scaled = scaler.transform(va)

    predictions = model.transform(scaled)
    print("Stampam predikcije!")
    predictions.printSchema()

    query = predictions.writeStream \
        .foreach(InfluxDBWriter()) \
        .start()
    
    query.awaitTermination()