from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
import os
import time
import datetime
from imspacv import CrossValidatorWithStratificationID


def main():   
    
    # user to specify: hyper-params
    treeGrid = [10, 500]
    depthGrid = [3, 15]
    # user to specify: input data location
    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/data_RFRegressionTemplate.csv"
    # read data    
    spark = SparkSession.builder.appName(__file__).getOrCreate()    
    data = spark.read.option("header", "true").option("inferSchema", "true").csv(dataFileName)
    # user to specify: original column names for stratification IDs in data
    outerFoldCol = "OuterFoldID"
    innerFoldCol = "InnerFoldID"
    # user to specify: original column names for predictors and output in data
    orgOutputCol = "Ozone"
    orgPredictorCols = data.columns[1:-2]
    # user to specify: the collective column name for all predictors
    collectivePredictorCol = "features"
    # user to specify: the column name for prediction
    predictionCol = "prediction"
    # user to specify: the output location on s3
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/" + st + "/"
    # user to specify the output location on master
    resultDir_master = "/home/lichao.wang/code/lichao/test/Results/" + st + "/"
    
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
    featureAssembledData.cache()
    
    
    # the model (pipeline)
    rf = RandomForestRegressor(featuresCol=collectivePredictorCol, 
                               labelCol=orgOutputCol, 
                               featureSubsetStrategy="sqrt", 
                               minInstancesPerNode=1)
    # lr = LinearRegression(maxIter=1e5, standardization=True, featuresCol=collectivePredictorCol, labelCol=orgOutputCol)
    evaluator = RegressionEvaluator(predictionCol=predictionCol, labelCol=orgOutputCol)
    paramGrid = ParamGridBuilder()\
               .addGrid(rf.maxDepth, depthGrid)\
               .addGrid(rf.numTrees, treeGrid)\
               .build()    
    
    # cross-evaluation
    nEvalFolds = len(set(listUniqueOutFoldIDs))        
    predictionsAllData = None
    
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3, 0777)
    if not os.path.exists(resultDir_master):
        os.makedirs(resultDir_master, 0777)
        
    for iFold in range(nEvalFolds):
        condition = featureAssembledData[outerFoldCol] == iFold
        testData = featureAssembledData.filter(condition)
        trainData = featureAssembledData.filter(~condition)
        
        validator = CrossValidatorWithStratificationID(\
                        estimator=rf, 
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
        cvMetrics = cvModel.avgMetrics
        cvMetricsFileName = resultDir_s3 + "cvMetricsFold" + str(iFold)
        cvMetrics.coalesce(4).write.csv(cvMetricsFileName, header="true")
        # save the hyper-parameters of the best model
        bestParams = validator.getBestModelParams()
        fileBestParams = open(resultDir_master + "bestParamsFold" + str(iFold) + ".txt", "w")
        fileBestParams.writelines(str(bestParams))
        fileBestParams.close()
        # # save coefficients of the best model
        # fileCoef = open(resultDir_master + "coefsFold" + str(iFold) + ".txt", "w")
        # fileCoef.writelines("Intercept: {}\n".format(str(cvModel.bestModel.intercept)))
        # fileCoef.writelines("Coefficients: {}\n".format(str(cvModel.bestModel.coefficients)))
        # fileCoef.close()        
    
    # save all predictions
    predictionsFileName = resultDir_s3 + "predictionsAllData"
    predictionsAllData.select(orgOutputCol, predictionCol).write.csv(predictionsFileName, header="true")
    # save rmse
    rmse = evaluator.evaluate(predictionsAllData)
    fileRMSE = open(resultDir_master + "rmse.txt", "w")
    fileRMSE.writelines("rmse: {}".format(rmse))
    fileRMSE.close()
    
    spark.stop()



if __name__ == "__main__":
    main()