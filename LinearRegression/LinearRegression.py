from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import rand
from math import exp
import numpy

def addID(dataset, number, npar, name):
    nPoses = dataset.count()
    npFoldIDsPos = np.array(list(range(number)) * np.ceil(float(nPoses) / number))
    # select the actual numbers of FoldIds matching the count of positive data points
    npFoldIDs = npFoldIDsPos[:nPoses]
    # Shuffle the foldIDs to give randomness
    np.random.shuffle(npFoldIDs)
    rddFoldIDs = sc.parallelize(npFoldIDs, npar).map(int)
    dfDataWithIndex = dataset.rdd.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "orgData")
    dfNewKeyWithIndex = rddFoldIDs.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "key")
    dfJoined = dfDataWithIndex.join(dfNewKeyWithIndex, "_2") \
        .select("orgData.matched_positive_id", 'key') \
        .withColumnRenamed('key', name) \
        .coalesce(npar)
    return dfJoined

def main():
    nEvalFolds = 5
    nValiFolds = 5
    seed = 1
    # hyper-params
    logLambdas = list(numpy.arange(-2,3,1))
    lambdas = map(exp, logLambdas)
    alphas = list(numpy.arange(0,1.01,0.1))

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
    randCol = "rand_num"    
    # evalRandColAppendedData = featureAssembledData.select("*", rand(seed).alias(randCol))
    evalRandColAppendedData = featureAssembledData.withColumn(randCol, rand(seed))
    
    lr = LinearRegression(maxIter=1e5, regParam=0.01, elasticNetParam=0.9, standardization=True, 
                          featuresCol="features", labelCol="y")
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y")
    paramGrid = ParamGridBuilder()\
               .addGrid(lr.regParam, lambdas)\
               .addGrid(lr.elasticNetParam, alphas)\
               .build()    
    
    h = 1.0 / nEvalFolds
    
    predictionsAllData = None
    for iFold in range(nEvalFolds):
        lb = iFold * h
        ub = (iFold + 1) * h
        condition = (evalRandColAppendedData[randCol] >= lb) & (evalRandColAppendedData[randCol] < ub)
        testData = evalRandColAppendedData.filter(condition)
        trainData = evalRandColAppendedData.filter(~condition)
        
        validator = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator)
        cvModel = validator.fit(trainData)
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
