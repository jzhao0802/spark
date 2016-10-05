from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator



def main():
    nEvalFolds = 5
    nValiFolds = 5
    seed = 1.0
    # hyper-params
    logLambdas = list(range(-2,3,1))
    lambdas = exp(logLambdas)
    alphas = list(range(0,1.01,0.1))

    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/data_linear_regression.csv"
    spark = SparkSession\
           .builder\
           .appName("Linear Regression")\
           .getOrCreate()
    
    data = spark\
          .read\
          .option("header", "true")\
          .option("inferSchema", "true")\
          .csv(dataFileName)
          
    assembler = VectorAssembler(inputCols=data.columns[1:], outputCol="features")
    featureAssembledData = assembler.transform(data).select("y", "features")    
    evalRandColAppendedData = featureAssembledData.select("*", rand(seed).alias(randCol))
    
    lr = LinearRegression(maxIter=1e5, regParam=0.01, elasticNetParam=0.9, standardization=True, 
                          featuresCol="features", labelCol="y")
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y")
    paramGrid = ParamGridBuilder()\
               .addGrid(lr.regParam, lambdas)\
               .addGrid(lr.elasticNetParam, alphas)\
               .build()    
    
    h = 1.0 / nEvalFolds
    randCol = "_rand"
    
    predictionsAllData = None
    for iFold in range(nEvalFolds):
        lb = iFold * h
        ub = (iFold + 1) * h
        condition = (evalRandColAppendedData[randCol] >= lb) & (evalRandColAppendedData[randCol] < ub)
        testData = evalRandColAppendedData.filter(condition)
        trainData = evalRandColAppendedData.filter(~condition)
        
        validator = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator)
        cvModel = cv.fit(trainData)
        predictions = cvModel.transform(testData)
        
        if predictionsAllData is not None:
            predictionsAllData = predictionsAllData.unionAll(predictions)
        else:
            predictionsAllData = predictions
    
    rmse = evaluator.evaluate(predictionsAllData)
    print("Final rmse: {}".format(rmse))
    
    spark.stop()

if __name__ == "__main__":
    main()
