from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import rand
from math import exp
import numpy
from imspacv import CrossValidatorWithStratificationID

def main():
    # user to specify: hyper-params
    lambdas = [0.1, 1]
    alphas = [0, 0.5]
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
    # user to specify: the output location on s3
    resultDir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/tmp/"    
    
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
    featureAssembledData = assembler.transform(data).select(orgOutputCol, collectivePredictorCol, outerFoldCol, innerFoldCol)    
    
    
    # the model (pipeline)
    lr = LinearRegression(maxIter=1e5, standardization=True, featuresCol=collectivePredictorCol, labelCol=orgOutputCol)
    evaluator = RegressionEvaluator(predictionCol=predictionCol, labelCol=orgOutputCol)
    paramGrid = ParamGridBuilder()\
               .addGrid(lr.regParam, lambdas)\
               .addGrid(lr.elasticNetParam, alphas)\
               .build()    
    
    # cross-evaluation
    nEvalFolds = len(set(listUniqueOutFoldIDs))        
    predictionsAllData = None
    
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3, 0777)
        
    for iFold in range(nEvalFolds):
        condition = featureAssembledData[outerFoldCol] == iFold
        testData = featureAssembledData.filter(condition)
        trainData = featureAssembledData.filter(~condition)
        
        validator = CrossValidatorWithStratificationID(\
                        estimator=lr, 
                        estimatorParamMaps=paramGrid, 
                        evaluator=evaluator, 
                        stratifyCol=innerFoldCol\
                    )
        cvModel = validator.fit(trainData)
        predictions = cvModel.transform(testData)
        
        if predictionsAllData is not None:
            predictionsAllData = predictionsAllData.unionAll(predictions)
        else:
            predictionsAllData = predictions
            
        # save the metrics for all hyper-parameter sets in cv
        cvMetrics = validator.getCVMetrics()
        cvMetrics.write.csv(resultDir_s3 + "cvMetricsFold" + str(iFold))
        # save the hyper-parameters of the best model
        bestParams = validator.getBestModelParams()
        fileBestParams = open(resultDir_s3 + "bestParamsFold" + str(iFold) + ".txt", "w")
        fileBestParams.writeLines(str(bestParams))
        fileBestParams.close()
        # save coefficients of the best model
        fileCoef = open(resultDir_s3 + "coefsFold" + str(iFold) + ".txt", "w")
        fileCoef.writelines("Intercept: {}".format(str(cvModel.bestModel.intercept)))
        fileCoef.writeLines("Coefficients: {}".format(str(cvModel.bestModel.coefficients)))
        fileCoef.close()        
    
    # save all predictions
    predictionsAllData.write.csv(resultDir_s3 + "predictionsAllData")
    # save rmse
    rmse = evaluator.evaluate(predictionsAllData)
    fileRMSE = open(resultDir_s3 + "rmse.txt", "w")
    fileRMSE.writelines("rmse: {}".format(rmse))
    fileRMSE.close()
    
    spark.stop()

if __name__ == "__main__":
    main()
