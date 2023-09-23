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

def Inicijalizacija():

    spark = SparkSession.builder.appName('Projekat3').master("spark://spark-master:7077").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    path = "data/denverVehiclesCleaned.csv"

    bus_df = spark.read.option("inferSchema", False).option("header", True).csv(HDFS_INPUT)

    df = bus_df.withColumn("datetime", concat(col("date"), lit(" "), col("time")))
    df = df.withColumn("datetime", to_timestamp("datetime", "MM-dd-yyyy HH:mm:ss"))
    df = df.drop("date")
    df = df.drop("time")

    # Generisanje labele
    #label = when(((df["speed"] < 15) & ((hour(df["datetime"]) >= 14) & (hour(df["datetime"]) < 18))) | (
            #(df["speed"] < 15) & ((hour(df["datetime"]) >= 7) & (hour(df["datetime"]) < 9))), 1).otherwise(0)
    #df = df.withColumn("guzva", label)

    df = df.withColumn("datetime", date_format("datetime", "MM-dd-yyyy HH:mm:ss"))

    df = df.withColumn("latitude", df["latitude"].cast(DoubleType()))
    df = df.withColumn("longitude", df["longitude"].cast(DoubleType()))
    df = df.withColumn("speed", df["speed"].cast(DoubleType()))

    # brisanje pogresnih vrednosti (outliera)
    df = df.filter(df.speed <= 120)
    df = df.filter(df.latitude >= -28)
    df = df.filter(df.latitude <= -18)

    df = df.filter(df.longitude >= -48)
    df = df.filter(df.longitude <= -38)
    return df


def KlasifikacijaLR(train, test):
    lr = LogisticRegression(featuresCol='scaled_features', labelCol="guzva", predictionCol='prediction',
                            maxIter=10, regParam=0.3, elasticNetParam=0.8)

    lrModel = lr.fit(train)
    predictions = lrModel.transform(test)
    predictions.select('guzva', 'prediction').show()

    lrModel.write().overwrite().save(MODEL_LOCATION)

    predictions.where((predictions.guzva == 1) & (predictions.prediction == 1)).show()
    print("Accuracy: {0}".format(lrModel.summary.accuracy))

    print("Kombinovana F mera: {0}".format(lrModel.summary.fMeasureByLabel))


def KlasifikacijaNB(train, test):
    nb = NaiveBayes(smoothing=1.0, featuresCol='scaled_features', labelCol="guzva", predictionCol='prediction')
    nbModel = nb.fit(train)
    predictions = nbModel.transform(test)
    predictions.select('guzva', 'prediction').show()

    nbModel.write().overwrite().save(MODEL_LOCATION)
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='guzva', metricName='accuracy')

    accuracy = evaluator.evaluate(predictions)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))


def Klasifikacija(train, test):
    dt = DecisionTreeClassifier(featuresCol='scaled_features', labelCol="guzva", predictionCol='prediction', maxDepth=5,
                                impurity="gini")
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)
    predictions.select('guzva', 'prediction').show()

    print("Accuracy: {0}".format(dtModel.summary.accuracy))

    print("Kombinovana F mera: {0}".format(dtModel.summary.fMeasureByLabel))


def Regresija(train, test):
    lr = (LinearRegression(featuresCol='scaled_features', labelCol="speed", predictionCol='prediction',
                           maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))
    linearModel = lr.fit(train)
    predictions = linearModel.transform(test)
    predictions.select('speed', 'prediction').show()

    linearModel.write().overwrite().save(MODEL_LOCATION)

    print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))

    print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))

    print("R2: {0}".format(linearModel.summary.r2))


if __name__ == '__main__':
    df = Inicijalizacija()
    # df = df.sample(fraction=1, seed=42)
    df.show(n=5)
    # print(df.count())
    # Brisanje redova koji sadrze null vrednosti
    # df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    df = df.dropna()

    # for col in df.columns:
    # distinct_values = df.select(col).distinct().count()
    # print(f"Kolona '{col}' ima {distinct_values} razlicitih vrednosti")

    indexer_model = StringIndexer(inputCols=['busID', 'busLine', 'datetime'],
                                  outputCols=['busID_num', 'busLine_num', 'datetime_num']).fit(df)

    indexer_model.write().overwrite().save(INDEXER_LOCATION)
    df_transformed = indexer_model.transform(df)

    df_transformed = df_transformed.drop('busID')
    df_transformed = df_transformed.drop('busLine')
    df_transformed = df_transformed.drop('datetime')

    df_transformed.show(n=5)

    columns = ["latitude", "longitude", "datetime_num", "busID_num", "busLine_num"]  # ovde treba i speed ako cemo klasifikaciju
    va = VectorAssembler().setInputCols(columns).setOutputCol('features').setHandleInvalid("skip").transform(
        df_transformed)
    va.describe().show()

    split = va.randomSplit([0.8, 0.2], 42)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(split[0])
    scaler_model.write().overwrite().save(SCALER_LOCATION)

    train = scaler_model.transform(split[0])
    test = scaler_model.transform(split[1])

    Regresija(train, test)

    #KlasifikacijaNB(train, test)