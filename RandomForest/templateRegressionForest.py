"""

Instructions: 

0. What the script does: This script serves as a template for using random regression forest and an outer cross-evaluation plus an inner cross-validation for hyper-parameter selection. Both the outer cross-evaluation loop and the inner cross-validation loop use predefined fold IDs included in the input data. Using the template, it is possible to obtain the following outputs: 
0.1. The rmse values of all hyper-parameter sets in every outer cross-evlauation round; 
0.2. The best hyper-parameter set in every outer cross-evlauation round; 
0.3. Variable importances of the forest models;
0.4. Predictions and ground-truth outputs for the entire input data (across all evaluation rounds); 
0.5. The overall rmse value for the entire input data. 

1. How to run the code: The template code depends on the module imspacv (currently in CrossValidator/imspacv.py). As an example command for submitting the script, run 

sudo spark-submit --deploy-mode client --master yarn --num-executors 5 --executor-cores 16 --executor-memory 19g --py-files /path/to/imspacv.py /path/to/templateRegressionForest.py

2. How to update the template for different specifications: 
2.1 In general, important fields to specify are listed at the beginning of the main function. Please refer to the comments in the code. The following is the details:
2.1.1. 1D grids for forest hyper-parameters: please specify separately the grids for the hyper-parameters of the forest. For the complete list of available hyper-parameters for RandomForestRegressor, please refer to https://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html 
2.1.2. The input datafile name: it needs to be a location on s3. The data needs to include the following information: an output column, one or more predictor columns and two columns of outer and inner stratification fold ID information. 
2.1.3. (Optional) You could specify the app name by relpacing the variable "__file__" with your preferred name. 
2.1.4. The column names of the outer and inner stratification fold IDs. The program uses them to find stratification information stored in the input data. 
2.1.5. The column name of the output / dependent variable. 
2.1.6. The column names of the predictors (as a list). 
2.1.7. The desired collective column name for all predictors, e.g., "features". 
2.1.8. The desired name for the prediction column, e.g., "prediction". 
2.1.9. The output location on both s3 and the master node, preferrably different everytime the program is run. This could be achieved by e.g., using the current timestamp as the folder name. 
2.2. Other fields to specify. The following is some possibilities but not an exhaustive list: 
2.2.1. Arguments for initialising RandomForestRegressor, RegressionEvaluator, etc. Please refer to https://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html for details. 
2.2.2. If one needs to overwrite an existing csv file on s3, add the argument mode="overwrite" when calling DataFrame.write.csv();
2.2.3. One could specify the file names for various outputs

"""



from pyspark.sql import SparkSession
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
    evaluator = RegressionEvaluator(predictionCol=predictionCol, labelCol=orgOutputCol)
    paramGrid = ParamGridBuilder()\
               .addGrid(rf.maxDepth, depthGrid)\
               .addGrid(rf.numTrees, treeGrid)\
               .build()    
    
    # cross-evaluation
    nEvalFolds = len(set(listUniqueOutFoldIDs))        
    predictionsAllData = None
    
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3, 0o777)
    if not os.path.exists(resultDir_master):
        os.makedirs(resultDir_master, 0o777)
        
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
        # save variable importance
        importances = cvModel.bestModel.featureImportances
        fileNameImportances = resultDir_master + "importanceFold" + str(iFold) + ".txt"
        with open(fileNameImportances, "w") as fileImportances:
            fileImportances.writelines(str(importances))
        os.chmod(fileNameImportances, 0o777)
        # save the hyper-parameters of the best model
        bestParams = validator.getBestModelParams()
        fileNameBestParams = resultDir_master + "bestParamsFold" + str(iFold) + ".txt"
        with open(fileNameBestParams, "w") as fileBestParams: 
            fileBestParams.writelines(str(bestParams))
        os.chmod(fileNameBestParams, 0o777)
             
    
    # save all predictions
    predictionsFileName = resultDir_s3 + "predictionsAllData"
    predictionsAllData.select(orgOutputCol, predictionCol).write.csv(predictionsFileName, header="true")
    # save rmse
    rmse = evaluator.evaluate(predictionsAllData)
    fileNameRMSE = resultDir_master + "rmse.txt"
    with open(fileNameRMSE, "w") as fileRMSE:
        fileRMSE.writelines("rmse: {}".format(rmse))
    os.chmod(fileNameRMSE, 0o777)
    
    spark.stop()

if __name__ == "__main__":
    main()