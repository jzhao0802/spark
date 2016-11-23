'''

Instructions:

0.  This script serves as a template for using bagging Random Forest model for classification with an outer cross-evaluation, a bootstrap aggregating (a.k.a bagging) algorithm for stability and accuracy improvement and an inner cross-validation for hyper-parameter selection
    The outer cross-evaluation, the bagging and the inner cross-validation loops all use predefined fold IDs included in the input data.
    Using the template, it is possible to obtain the following outputs:
    0.1. The AUC and AUPR values of all hyper-parameter sets in every inner cross-validation round;
    0.2. The best hyper-parameter set in every bagging round;
    0.3. Average predictions(probability of label 1) across bagging and true label for the entire input data (across all outer cross-evaluation rounds);
    0.4. The overall AUC and AUPR values for the entire input data.

1.  How to run the code: The template code depends on the module imspacv (currently in CrossValidator/imspacv.py). To run the script, do the following:
    1.1. Put both the template script and imspacv.py under the same location;
    1.2. Change your current location to the location in 1.1.
    1.3. Execute in the command line: sudo spark-submit --deploy-mode client --master yarn --num-executors 5 --executor-cores 16 --executor-memory 19g --py-files imspacv.py templateBaggingForest.py

2.  How to update the template for different specifications:
    2.1 In general, important fields to specify are listed at the beginning of the main function. Please refer to the comments in the code.
        The following is the details:
        2.1.1.  1D grids for hyper-parameters for individual forests, such as the number of trees, depth and nodesize, etc.
                Please give one list to every hyper-parameter you'd like to search along.
        2.1.2   Seed will be used in Random Forest function.
        2.1.3   Original column name for outcome, the outcome should be either interger or double type.
        2.1.4.  The input datafile name:
                It needs to be a location on s3.
                The data needs to include the following information:
                    i. an output column,
                    ii. one or more predictor columns,
                    iii. the primary key used for computing the average prediction, it's unique for each line,
                    iv. three columns of outer, bagging and inner stratification fold ID information. The bagging stratification fold ID is for stratifying data for each forest.
        2.1.5.  (Optional) You could specify the app name by replacing the variable "__file__" with your preferred name.
        2.1.6.  The column names of the outer, bagging and inner stratification fold IDs. The program uses them to find stratification information stored in the input data.
        2.1.7.  The column name of primary key.
        2.1.8.  The column name of the output / dependent variable.
        2.1.9.  The column names of the predictors (as a list).
        2.1.10.  The desired collective column name for all predictors, e.g. "features".
        2.1.11.  The name of the selected prediction column from individual Random Forest models (predictions from all forest will be concatenated into one DataFrame).
        2.1.12.  The desired name of the predicted probability of label 1.
        2.1.13.  The desired name of the averaged probability of label 1 across bagging.
        2.1.14.  The output location on both s3 and the master node,preferably different every time the program is run.
                This could be achieved by e.g., using the current timestamp as the folder name.

    2.2 Other fields to specify. The following is some possibilities but not an exhaustive list:
        2.2.1.  Arguments for other functions such as initialising Random Forest, etc. Please refer to
                https://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html for details.
        2.2.2.  If one needs to overwrite an existing csv file on s3, add the argument mode="overwrite" when calling DataFrame.write.csv();
        2.2.3.  One could specify the file names for various outputs


'''

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
import time
import datetime
from imspacv import CrossValidatorWithStratificationID

# function: get probability with label=1 from probability vector
def getitem(i):
    def getitem_(v):
         return v.array.item(i)
    return udf(getitem_, DoubleType())

def main():
    # user to specify: hyper-params for individual forests
    numtree = [20, 30]
    numdepth = [3, 4]
    nodesize = [3, 5]
    mtry = ["auto", "2"]
    # user to specify : seed in Random Forest model
    iseed = 42
    # user to specify: input data location
    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/Hui/template_test/data/data_baggingRFTemplat_FoldID.csv"
    # read data
    spark = SparkSession.builder.appName(__file__).getOrCreate()
    data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(dataFileName)
    # user to specify: original column names for stratification IDs in data
    outerFoldCol = "OuterFoldID"
    innerFoldCol = "InnerFoldID"
    baggingCol = "BaggingID"
    # user to sepecify: primary key in data
    primaryKey = "patid"
    # user to specify: original column names for predictors and outcome in data
    orgOutputCol = "label"
    orgPredictorCols = data.columns[1:-5]
    # sanity check
    if type(data.select(orgOutputCol).schema.fields[0].dataType) not in (DoubleType, IntegerType):
        raise TypeError("The output column is not of type integer or double. ")
    data = data.withColumn(orgOutputCol, data[orgOutputCol].cast("double"))
    # user to specify: the name of the selected prediction column from Random Forest model.
    selPredictionColFromModel = "probability"
    # user to specify: the desired collective column name for all predictors
    collectivePredictorCol = "features"
    # user to specify: the desired name of the predicted probability of label 1.
    predictedProbLabel1 = "prob_1"
    # user to specify: the desired name of the averaging probability of label 1 across bagging.
    avgPredictedProbLabel1 = "avg_prob_1"
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
    if baggingCol not in data.columns:
        raise ValueError("baggingCol " + baggingCol + " doesn't exist in the input data. ")
    rowsUniqueOutFoldIDs = data.select(outerFoldCol).distinct().collect()
    listUniqueOutFoldIDs = [x[outerFoldCol] for x in rowsUniqueOutFoldIDs]
    if (set(listUniqueOutFoldIDs) != set(range(max(listUniqueOutFoldIDs)+1))):
        raise ValueError("The outerFoldCol column " + outerFoldCol +
                         " does not have zero-based consecutive integers as fold IDs.")
    rowsUniqueBaggingIDs = data.select(baggingCol).distinct().collect()
    listUniqueBaggingIDs = [x[baggingCol] for x in rowsUniqueBaggingIDs]
    if (set(listUniqueBaggingIDs) != set(range(max(listUniqueBaggingIDs)+1))):
        raise ValueError("The BaggingCol column " + baggingCol +
                         " does not have zero-based consecutive integers as fold IDs.")
    if primaryKey not in data.columns:
        raise ValueError("primaryKey " + primaryKey + " doesn't exist in the input data. ")


    # convert to ml-compatible format
    assembler = VectorAssembler(inputCols=orgPredictorCols, outputCol=collectivePredictorCol)
    featureAssembledData = assembler.transform(data)\
        .select(orgOutputCol, collectivePredictorCol, outerFoldCol,
                innerFoldCol, baggingCol, primaryKey)
    featureAssembledData.cache()

    # the individual forest model (pipeline)
    rf = RandomForestClassifier(featuresCol = collectivePredictorCol,
                                labelCol = orgOutputCol, seed=iseed)

    evalForGrid = BinaryClassificationEvaluator(rawPredictionCol=selPredictionColFromModel,
                                              labelCol=orgOutputCol)
    evalForAllData = BinaryClassificationEvaluator(rawPredictionCol=avgPredictedProbLabel1,
                                              labelCol=orgOutputCol)

    paramGrid = ParamGridBuilder()\
            .addGrid(rf.numTrees, numtree)\
            .addGrid(rf.maxDepth, numdepth)\
            .addGrid(rf.minInstancesPerNode, nodesize)\
            .addGrid(rf.featureSubsetStrategy, mtry)\
            .build()

    validator = CrossValidatorWithStratificationID(\
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evalForGrid,
        stratifyCol=innerFoldCol
        )

    # outer cross-evaluation
    nEvalFolds = len(set(listUniqueOutFoldIDs))
    nEvalBags = len(set(listUniqueBaggingIDs))
    predictionsAllData = None

    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3, 0o777)
    if not os.path.exists(resultDir_master):
        os.makedirs(resultDir_master, 0o777)
    os.chmod(resultDir_master, 0o777)

    for iFold in range(nEvalFolds):
        condition = featureAssembledData[outerFoldCol] == iFold
        testData = featureAssembledData.filter(condition)
        trainData = featureAssembledData.filter(~condition)

        # split trainData into positive and negative
        posTrainData = trainData.filter(trainData[orgOutputCol] == 1)
        negTrainData = trainData.filter(trainData[orgOutputCol] == 0)

        predictionsBagData = None

        # bagging cross-evaluation
        for iBag in range(nEvalBags):
            bagCondition = negTrainData[baggingCol] == iBag
            negTrainDataThisBag = negTrainData.filter(bagCondition)

            # combine all the posTrainData with partial iNegTrainData
            bagTrainData = posTrainData.unionAll(negTrainDataThisBag)

            cvModel = validator.fit(bagTrainData)
            predictions = cvModel.transform(testData)

            baggingPrediction = predictions\
                .select(primaryKey, orgOutputCol,
                        getitem(1)(selPredictionColFromModel).alias(predictedProbLabel1))

            if predictionsBagData is not None:
                predictionsBagData = predictionsBagData.unionAll(baggingPrediction)
            else:
                predictionsBagData = baggingPrediction

            # save the metrics for all hyper-parameter sets in cv
            cvMetrics = cvModel.avgMetrics
            cvMetricsFileName = resultDir_s3 + "cvMetricsFold" + str(iFold) + "Bagging" + str(iBag)
            cvMetrics.coalesce(4).write.csv(cvMetricsFileName, header="true")

            # save the hyper-parameters of the best model
            bestParams = validator.getBestModelParams()
            with open(resultDir_master + "bestParamsFold" + str(iFold) +
                              "Bagging"+ str(iBag) +".txt","w") as fileBestParams:
                fileBestParams.write(str(bestParams))
            os.chmod(resultDir_master + "bestParamsFold" + str(iFold) +
                     "Bagging" + str(iBag) +".txt", 0o777)
            # save importance score of the best model
            with open(resultDir_master + "importanceScoreFold" + str(iFold) +
                              "Bagging" + str(iBag) + ".txt","w") as filecvCoef:
                for id in range(len(orgPredictorCols)):
                    filecvCoef.write("{0} : {1}".format(orgPredictorCols[id], cvModel.bestModel.featureImportances[id]))
                    filecvCoef.write("\n")
            os.chmod(resultDir_master + "importanceScoreFold" + str(iFold) +
                     "Bagging" + str(iBag) + ".txt", 0o777)

        # averaging predictions across bagging by the primaryKey
        avgPrediction = predictionsBagData.groupBy(primaryKey)\
            .avg(orgOutputCol,predictedProbLabel1)\
            .withColumnRenamed("avg(" + orgOutputCol + ")", orgOutputCol)\
            .withColumnRenamed("avg(" + predictedProbLabel1 + ")", avgPredictedProbLabel1)

        # output the performance for each outer loop
        auc_ot = evalForAllData.evaluate(avgPrediction,
                                 {evalForAllData.metricName:"areaUnderROC"})
        aupr_ot = evalForAllData.evaluate(avgPrediction,
                                  {evalForAllData.metricName:"areaUnderPR"})
        with open(resultDir_master + "auc_aupr_outerFold"
                          + str(iFold) + ".txt", "w") as filePerf:
            filePerf.write("AUC: {}".format(auc_ot))
            filePerf.write('\n')
            filePerf.write("AUPR: {}".format(aupr_ot))
        os.chmod(resultDir_master + "auc_aupr_outerFold"
                          + str(iFold) + ".txt", 0o777)

        # store the average predictions
        if predictionsAllData is not None:
            predictionsAllData = predictionsAllData.unionAll(avgPrediction)
        else:
            predictionsAllData = avgPrediction

    # save all predictions
    predictionsFileName = resultDir_s3 + "predictionsAllData"
    predictionsAllData.select(orgOutputCol, avgPredictedProbLabel1)\
        .write.csv(predictionsFileName, header="true")

    # save AUC and AUPR
    auc = evalForAllData.evaluate(predictionsAllData,
                                  {evalForAllData.metricName:"areaUnderROC"})
    aupr = evalForAllData.evaluate(predictionsAllData,
                                   {evalForAllData.metricName:"areaUnderPR"})

    with open(resultDir_master + "auc_aupr.txt", "w") as filePerf:
        filePerf.write("AUC: {}".format(auc))
        filePerf.write('\n')
        filePerf.write("AUPR: {}".format(aupr))
    os.chmod(resultDir_master + "auc_aupr.txt", 0o777)

    spark.stop()

if __name__ == "__main__":
    main()
