'''

Instructions:

0.  This script serves as a template for using logistic regression with the
    elastic-net penalty and an outer plus an inner cross-validation for hyper-parameter selection.
    Both the outer and the inner cross-validation loops uses predefined fold IDs included in the input data.

    Using the template, it is possible to obtain the following outputs:
        0.1. The AUC values of all hyper-parameter sets in every outer cross-evaluation round;
        0.2. The best hyper-parameter set in every outer cross-evaluation round;
        0.3. Predictions(probability of label 1) and true label for the entire input data (across all evaluation rounds);
        0.4. The overall AUC and AUPR values for the entire input data.

1.  How to run the code: The template code depends on the module imspacv (currently in CrossValidator/imspacv.py).
    As an example command for submitting the script, run

    sudo spark-submit --deploy-mode client --master yarn --num-executors 5
    --executor-cores 16 --executor-memory 19g --py-files /path/to/imspacv.py
    /path/to/LogisticRegression_EN.py

2.  How to update the template for different specifications:
    2.1 In general, important fields to specify are listed at the beginning of the main function.
        Please refer to the comments in the code. The following is the details:
        2.1.1.  1D grids for lambda and alpha:
                Please specify separately the grids for the hyper-parameters lambda and alpha.
                Lambda is the overall weight of the regularizer (>0) and alpha controls the relative weights between the L1 and L2 norms (0<=alpha<=1)
                Alpha=1 it is equivalent to a LASSO model, alpha=0 it is a ridge model.
                Please specify each grid as a list.
        2.1.2.  The input datafile name:
                It needs to be a location on s3.
                The data needs to include the following information:
                    i. an output column,
                    ii. one or more predictor columns
                    iii. two columns of outer and inner stratification fold ID information.
        2.1.3.  (Optional) You could specify the app name by relpacing the variable "__file__" with your preferred name.
        2.1.4.  The column names of the outer and inner stratification fold IDs.
                The program uses them to find stratification information stored in the input data.
        2.1.5.  The column name of the output / dependent variable.
        2.1.6.  The column names of the predictors (as a list).
        2.1.7.  The desired collective column name for all predictors, e.g. "features".
        2.1.8.  The desired name for the prediction column, e.g., "probability".
        2.1.9.  The output location on both s3 and the master node, preferably different every time the program is run.
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
from imspacv import CrossValidatorWithStratificationID

def main():
    # user to specify: hyper-params
    lambdas = [0.1, 1]
    alphas = [0, 0.5]
    # user to specify: input data location
    dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/Hui/template_test/data/data_LRRFTemplat_FoldID.csv"
    # read data
    spark = SparkSession.builder.appName(__file__).getOrCreate()
    data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(dataFileName)
    # user to specify: original column names for stratification IDs in data
    outerFoldCol = "OuterFoldID"
    innerFoldCol = "InnerFoldID"
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
                            labelCol = orgOutputCol, standardization = True)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol=predictionCol,
                                              labelCol=orgOutputCol)
    paramGrid = ParamGridBuilder()\
               .addGrid(lr.regParam, lambdas)\
               .addGrid(lr.elasticNetParam, alphas)\
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
        cvMetrics = cvModel.avgMetrics
        cvMetricsFileName = resultDir_s3 + "cvMetricsFold" + str(iFold)
        cvMetrics.coalesce(1).write.csv(cvMetricsFileName, header="true")
        # save the hyper-parameters of the best model
        bestParams = validator.getBestModelParams()
        with open(resultDir_master + "bestParamsFold" + str(iFold) + ".txt", "w") as fileBestParams:
            fileBestParams.write(str(bestParams))
        # save coefficients of the best model
        with open(resultDir_master + "coefsFold" + str(iFold) + ".txt","w") as filecvCoef:
            filecvCoef.write("Intercept: {}".format(str(cvModel.bestModel.intercept)))
            filecvCoef.write("\n")
            for id in range(len(orgPredictorCols)):
                filecvCoef.write("%s : %f" %(orgPredictorCols[id], cvModel.bestModel.coefficients[id]))
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



