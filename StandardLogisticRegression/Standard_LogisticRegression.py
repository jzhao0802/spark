'''

Instructions:

0.  This script serves as a template for using standard logistic regression with an outer cross-validation.
    The outer cross-validation loops uses predefined fold IDs included in the input data.

    Using the template, it is possible to obtain the following outputs:
        0.1. The AUC and AUPR values in every outer cross-evaluation round;
        0.2. Predictions(probability of label 1) and true label for the entire input data (across all evaluation rounds);
        0.3. The overall AUC and AUPR values for the entire input data.

1.  How to run the code:
    As an example command for submitting the script, run

    sudo spark-submit --deploy-mode client --master yarn --num-executors 5
    --executor-cores 16 --executor-memory 19g /path/to/LogisticRegression_EN.py

2.  How to update the template for different specifications:
    2.1 In general, important fields to specify are listed at the beginning of the main function.
        Please refer to the comments in the code. The following is the details:
        2.1.1.  The input datafile name:
                It needs to be a location on s3.
                The data needs to include the following information:
                    i. an output column,
                    ii. one or more predictor columns
                    iii. one columns of outer stratification fold ID information.
        2.1.2.  (Optional) You could specify the app name by relpacing the variable "__file__" with your preferred name.
        2.1.3.  The column names of the outer stratification fold IDs.
                The program uses it to find stratification information stored in the input data.
        2.1.4.  The column name of the output / dependent variable.
        2.1.5.  The column names of the predictors (as a list).
        2.1.6.  The desired collective column name for all predictors, e.g. "features".
        2.1.7.  The desired name for the prediction column, e.g., "probability".
        2.1.8.  The output location on both s3 and the master node, preferably different every time the program is run.
                This could be achieved by e.g., using the current timestamp as the folder name.

    2.2 Other fields to specify. The following is some possibilities but not an exhaustive list:
        2.2.1.  If one needs to overwrite an existing csv file on s3, add the argument mode="overwrite" when calling DataFrame.write.csv();
        2.2.2.  One could specify the file names for various outputs


'''

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy
import os
import time
import datetime


def main():
    # user to specify: input data location
    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/Hui/template_test/data/data_LRRFTemplat_FoldID.csv"
    # read data
    spark = SparkSession.builder.appName(__file__).getOrCreate()
    data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(dataFileName)
    # user to specify: original column names for stratification IDs in data
    outerFoldCol = "OuterFoldID"
    # user to specify: original column names for predictors and output in data
    orgOutputCol = "y"
    orgPredictorCols = data.columns[1:-2]
    # user to specify: the collective column name for all predictors
    collectivePredictorCol = "features"
    # user to specify: the column name for prediction
    predictionCol = "probability"
    # user to specify: the output location on s3
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/Hui/template_test/results/" + st + "/"
    # user to specify the output location on master
    resultDir_master = "/home/hjin/template_test/results/" + st + "/"

    # sanity check
    if outerFoldCol not in data.columns:
        raise ValueError("outerFoldCol " + outerFoldCol + " doesn't exist in the input data. ")
    if innerFoldCol not in data.columns:
        raise ValueError("innerFoldCol " + innerFoldCol + " doesn't exist in the input data. ")
    rowsUniqueOutFoldIDs = data.select(outerFoldCol).distinct().collect()
    listUniqueOutFoldIDs = [x[outerFoldCol] for x in rowsUniqueOutFoldIDs]
    if (set(listUniqueOutFoldIDs) != set(range(max(listUniqueOutFoldIDs)+1))):
        raise ValueError("The outerFoldCol column " + outerFoldCol +
                         " does not have zero-based consecutive integers as fold IDs.")

    # convert to ml-compatible format
    assembler = VectorAssembler(inputCols=orgPredictorCols, outputCol=collectivePredictorCol)
    featureAssembledData = assembler.transform(data)\
        .select(orgOutputCol, collectivePredictorCol, outerFoldCol, innerFoldCol)


    # the model (pipeline)
    lr = LogisticRegression(maxIter=1e5, featuresCol = collectivePredictorCol,
                            labelCol = orgOutputCol, regParam=0.0,
                            elasticNetParam=0.0, standardization = True)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol=predictionCol,
                                              labelCol=orgOutputCol)

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

        cvModel = lr.fit(trainData)

        predictions = cvModel.transform(testData)

        if predictionsAllData is not None:
            predictionsAllData = predictionsAllData.unionAll(predictions)
        else:
            predictionsAllData = predictions

        # save the metrics for all hyper-parameter sets in cv
        cvauc = evaluator.evaluate(predictions,
                             {evaluator.metricName:"areaUnderROC"})
        cvaupr = evaluator.evaluate(predictions,
                              {evaluator.metricName:"areaUnderPR"})
        with open(resultDir_master + "cvMetricsFold" + str(iFold) + ".txt", "w") as filecvPerf:
            filecvPerf.write("cvAUC: {}".format(cvauc))
            filecvPerf.write('\n')
            filecvPerf.write("cvAUPR: {}".format(cvaupr))

        # save coefficients of the best model
        with open(resultDir_master + "coefsFold" + str(iFold) + ".txt", "w") as filecvCoef:
            filecvCoef.write("Intercept: {}".format(str(cvModel.intercept)))
            filecvCoef.write("\n")
            for id in range(len(orgPredictorCols)):
                filecvCoef.write("%s : %f" %(orgPredictorCols[id], cvModel.coefficients[id]))
                filecvCoef.write("\n")

    # save all predictions
    predictionsFileName = resultDir_s3 + "predictionsAllData"
    predictionsAllData.select(orgOutputCol, predictionCol)\
        .write.csv(predictionsFileName, header="true")
    # save AUC and AUPR
    auc = evaluator.evaluate(predictionsAllData,
                             {evaluator.metricName:"areaUnderROC"})
    aupr = evaluator.evaluate(predictionsAllData,
                              {evaluator.metricName:"areaUnderPR"})
    with open(resultDir_master + "auc_aupr.txt", "w") as filePerf:
        filePerf.write("AUC: {}".format(auc))
        filePerf.write('\n')
        filePerf.write("AUPR: {}".format(aupr))

    spark.stop()

if __name__ == "__main__":
    main()



