"""
Created by Hui on 27/4/2016

This function could run Random Forest with Spark-submit using Pipeline in Spark ML module.
argv 1 = app name
argv 2 = inpath, '/200k/' or '/3m/'
argv 3 = start of numTree seq,
argv 4 = end of numTree seq,
argv 5 = number of numTree ,
argv 6 = start of maxDepth seq,
argv 7 = end of maxDepth seq,
argv 8 = number of maxDepth,
argv 9 = number of folds in CV
argv 10 = number of partitions

"""

# Import the packages
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
import numpy as np
#import random

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Some constance
data_path = "s3://emr-rwes-pa-spark-dev-datastore/Shire_test"
tr_file = 'tr.csv'
ts_file = 'ts.csv'
app_name = 'RF_Hyper_CV_simple'
inpath = 'Shire_test/200k_patid/'
start_tree = int(100)
stop_tree = int(300)
num_tree = int(3)
start_depth = int(1)
stop_depth = int(5)
num_depth = int(5)
fold = int(3)
par = int(50)
CVCriteria = 'areaUnderROC' #areaUnderROC or areaUnderPR

# Define the functions
def load_csv_file(sc, data_path, inpath, file, par):

    #reading in data as RDD data
    data1 = sc.textFile(data_path + inpath + file, par) #in RDD format

    # Skip the header, and re-structure the columns as "label" and "features"
    header = data1.first()
    data2 = data1.filter(lambda x: x != header).map(lambda line: line.split(','))

    #column 0 is patient id, column 1 is label, from column 2 to end are features
    data3 = data2.map(lambda line: (line[0], line[1], Vectors.dense(np.asarray(line[2:]).astype(np.float32))))

    # Convert to Spark DataFrame
    data_df = sqlContext.createDataFrame(data3, ['patid', 'label', 'features'])

    # Convert label to double type
    return data_df.withColumn('label', data_df['label'].cast('double'))

def toCSVLine(data):
    return ','.join(str(d) for d in data)

def main(sc, app_name=app_name, data_path=data_path, inpath=inpath, tr_file=tr_file,
         ts_file=ts_file, start_tree=start_tree, stop_tree=stop_tree,
         num_tree=num_tree, start_depth=start_depth, stop_depth=stop_depth,
         num_depth=num_depth, fold=fold, par=par):

    #reading in data
    tr = load_csv_file(sc, data_path, inpath, tr_file, par)
    ts = load_csv_file(sc, data_path, inpath, ts_file, par)

    #transfer to RF invalid label column
    stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
    si_model = stringIndexer.fit(tr)
    tr_td = si_model.transform(tr)

    # Build the model
    rf = RandomForestClassifier(labelCol="indexed",seed=42)

    # Create the pipeline
    pipeline = Pipeline(stages=[rf])

    # Create the parameter grid builder
    paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, list(np.linspace(start_tree, stop_tree,
                                               num_tree).astype('int')))\
        .addGrid(rf.numTrees, list(np.linspace(start_depth, stop_depth,
                                               num_depth).astype('int')))\
        .build()


    # Create the evaluator
    evaluator = BinaryClassificationEvaluator(metricName=CVCriteria)

    #Create the cross validator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=fold)

    #run cross-validation and choose the best parameters
    cvModel = crossval.fit(tr_td)

    # Predict on training data
    prediction_tr = cvModel.transform(tr)
    #prediction_tr2 = prediction_tr.withColumn('label', prediction_tr[
    # 'label'].cast('double'))#convert label to double
    pred_score_tr = prediction_tr.select('patid', 'label', 'probability')

    #predict on test data
    prediction_ts = cvModel.transform(ts)
    #prediction_ts2 = prediction_ts.withColumn('label', prediction_ts[
    # 'label'].cast('double'))#convert label to double
    pred_score_ts = prediction_ts.select('patid', 'label', 'probability')

    # AUC
    AUC_tr = evaluator.evaluate(prediction_tr,{evaluator.metricName:CVCriteria} )
    AUC_ts = evaluator.evaluate(prediction_ts,{evaluator.metricName:CVCriteria} )

    #print out results
    print("Traing AUC = %s " % AUC_tr)
    print("Test AUC = %s " % AUC_ts)

    #output results to S3
    tr_lines = pred_score_tr.map(toCSVLine)
    tr_lines.saveAsTextFile((data_path + '/results/LR/' + app_name + '_pred_tr.csv' ))

    ts_lines = pred_score_ts.map(toCSVLine)
    ts_lines.saveAsTextFile((data_path + '/results/LR/' + app_name + '_pred_ts.csv'))



if __name__ == "__main__":

    sc = SparkContext(appName = app_name)

    sqlContext = SQLContext(sc)

    main(sc)

    sc.stop()



