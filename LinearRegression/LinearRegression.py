from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import rand
from math import exp
import numpy

def main():
    # user to specify: hyper-params
    logLambdas = list(numpy.arange(-2,3,1))
    lambdas = map(exp, logLambdas)
    alphas = list(numpy.arange(0,1.01,0.1))
    # user to specify: input data location
    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/data_LinearRegressionTemplate.csv"
    # read data    
    spark = SparkSession.builder.appName(__file__).getOrCreate()    
    data = spark.read.option("header", "true").option("inferSchema", "true").csv(dataFileName)
    # user to specify: original column names for stratification IDs in data
    outerFoldCol = "OuterFoldID"
    innerFoldCol = "InnerFoldID"
    # user to specify: original column names for predictors and output in data
    orgOutputCol = "y"
    orgPredictorCols = data.columns[1:-2]
    # user to specify: the collective column name for all predictors
    collectivePredictorCol = "features"
    # user to specify: the column name for prediction
    predictionCol = "prediction"
    
    # sanity check
    if outerFoldCol not in data.columns:
        raise ValueError("outerFoldCol " + outerFoldCol + " doesn't exist in the input data. ")
    if innerFoldCol not in data.columns: 
        raise ValueError("innerFoldCol " + innerFoldCol + " doesn't exist in the input data. ")
    rowsUniqueOutFoldIDs = data.select(outerFoldCol).distinct().collect()
    listUniqueOutFoldIDs = [x[outerFoldCol] for x in rowsUniqueOutFoldIDs]
    if (set(listUniqueOutFoldIDs) != set(range(max(listUniqueOutFoldIDs)+1))):
        raise ValueError("The outerFoldCol column " + outerFoldCol + " does not have zero-based consecutive integers as fold IDs.")
     
    
    # convert to ml-compatible format
    assembler = VectorAssembler(inputCols=orgPredictorCols, outputCol=collectivePredictorCol)
    featureAssembledData = assembler.transform(data).select(orgOutputCol, collectivePredictorCol)    
    
    
    # the model (pipeline)
    lr = LinearRegression(maxIter=1e5, standardization=True, featuresCol=collectivePredictorCol, labelCol=orgOutputCol)
    evaluator = RegressionEvaluator(predictionCol=predictionCol, labelCol=orgOutputCol)
    paramGrid = ParamGridBuilder()\
               .addGrid(lr.regParam, lambdas)\
               .addGrid(lr.elasticNetParam, alphas)\
               .build()    
    
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
