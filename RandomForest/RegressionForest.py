from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler


def main():
    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/data_RFRegressionTemplate.csv"
    spark = SparkSession.builder.appName(__file__).getOrCreate()    
    data = spark.read.option("header", "true").option("inferSchema", "true").csv(dataFileName)
    
    
    # # Automatically identify categorical features, and index them.
    # # Set maxCategories so features with > 4 distinct values are treated as continuous.
    # featureIndexer =\
        # VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # # Split the data into training and test sets (30% held out for testing)
    # (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    orgPredictorCols = data.columns[1:]
    collectivePredictorCol = "features"
    orgOutputCol = "Ozone"
    assembler = VectorAssembler(inputCols=orgPredictorCols, outputCol=collectivePredictorCol)
    featureAssembledData = assembler.transform(data).select(orgOutputCol, collectivePredictorCol)    

    # Train a RandomForest model.
    rf = RandomForestRegressor(featuresCol=collectivePredictorCol, 
                               labelCol=orgOutputCol, 
                               featureSubsetStrategy="sqrt", 
                               maxDepth=10,
                               minInstancesPerNode=1, 
                               numTrees=500)
    model = rf.fit(featureAssembledData)

    # Make predictions.
    predictions = model.transform(featureAssembledData)

    # # Select example rows to display.
    # predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    predictionCol = "prediction"
    evaluator = RegressionEvaluator(
        labelCol=orgOutputCol, predictionCol=predictionCol)
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE)".format(rmse))
    
    
    spark.stop()



if __name__ == "__main__":
    main()