from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator



def main():
    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/data_linear_regression.csv"
    spark = SparkSession\
           .builder\
           .appName("Linear Regression")\
           .getOrCreate()
    
    # schema = StructType([StructField("y", DoubleType()),
                         # StructField("V1", DoubleType()),
                         # StructField("C", StringType())])
    data = spark\
          .read\
          .option("header", "true")\
          .option("inferSchema", "true")\
          .csv(dataFileName)
          
    assembler = VectorAssembler(inputCols=data.columns[1:], outputCol="features")
    featureAssembledData = assembler.transform(data).select("y", "features")
    lr = LinearRegression(maxIter=1e5, regParam=0.01, elasticNetParam=0.9, standardization=True, 
                          featuresCol="features", labelCol="y")
    
    model = lr.fit(featureAssembledData)

    # # Print the coefficients and intercept for linear regression
    # print("Coefficients: " + str(model.coefficients))
    # print("Intercept: " + str(model.intercept))
    
    prediction = model.transform(featureAssembledData)
    
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y")
    metric = evaluator.evaluate(prediction)
    
    spark.stop()

if __name__ == "__main__":
    main()
