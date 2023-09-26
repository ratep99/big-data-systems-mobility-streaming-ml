import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, min, max, col, when, count
from pyspark.sql.functions import concat, date_format, lit, to_timestamp, hour
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, NaiveBayes, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import numpy as np

HDFS_INPUT = os.getenv('HDFS_BUS')
MODEL_LOCATION = os.getenv('MODEL_LOCATION')
SCALER_LOCATION = os.getenv('SCALER_LOCATION')
INDEXER_LOCATION=os.getenv('INDEXER_LOCATION')


def ZastojPrediction(train, test):
    naive_bayes = NaiveBayes(smoothing=1.0, featuresCol='scaled_features', labelCol="zastoj", predictionCol='prediction')
    model = naive_bayes.fit(train)
    predictions = model.transform(test)
    predictions.select('zastoj', 'prediction').show()
    model.write().overwrite().save(MODEL_LOCATION)
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='zastoj', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))


def Regression(train, test):
    lr = (LinearRegression(featuresCol='scaled_features', labelCol="speed_kmh", predictionCol='prediction',
                           maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))
    linearModel = lr.fit(train)
    predictions = linearModel.transform(test)
    predictions.select('speed_kmh', 'prediction').show()
    linearModel.write().overwrite().save(MODEL_LOCATION)
    print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))
    print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))
    print("R2: {0}".format(linearModel.summary.r2))


def load_data(spark):
    path = "data/denverVehiclesCleaned.csv"
    vehicles_df = spark.read.option("inferSchema", False).option("header", True).csv(HDFS_INPUT)
    return vehicles_df;


def process_data(dataframe):
    df = dataframe.withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"))
    label = when((df["speed_kmh"] < 15), 1).otherwise(0)
    df = df.withColumn("zastoj", label)
    df = df.withColumn("timestamp", date_format("timestamp", "MM-dd-yyyy HH:mm:ss"))
    df = df.withColumn("latitude", df["latitude"].cast(DoubleType()))
    df = df.withColumn("longitude", df["longitude"].cast(DoubleType()))
    df = df.withColumn("speed_kmh", df["speed_kmh"].cast(DoubleType()))
    return df


def Inicijalizacija():
    spark = SparkSession.builder.appName('Projekat3').master("spark://spark-master:7077").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = load_data(spark)
    dataframe = process_data(df)
    return dataframe

if __name__ == '__main__':
    df = Inicijalizacija()
    df.show(n=5)
    df = df.dropna()

    indexer_model = StringIndexer(inputCols=['id', 'timestamp', 'type'],
                                  outputCols=['id_num', 'timestamp_num', 'type_num']).fit(df)
    indexer_model.write().overwrite().save(INDEXER_LOCATION)

    df_transformed = indexer_model.transform(df)
    df_transformed = df_transformed.drop('id')
    df_transformed = df_transformed.drop('timestamp')
    df_transformed = df_transformed.drop('type')

    df_transformed.show(n=5)

    columns = ["latitude", "longitude", "timestamp_num", "id_num", "type_num"]
    va = VectorAssembler().setInputCols(columns).setOutputCol('features').setHandleInvalid("skip").transform(
        df_transformed)
    va.describe().show()

    split = va.randomSplit([0.8, 0.2], 42)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(split[0])
    scaler_model.write().overwrite().save(SCALER_LOCATION)

    train = scaler_model.transform(split[0])
    test = scaler_model.transform(split[1])

    Regression(train, test)
