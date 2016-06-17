__author__ = 'hjin'
__project__ = 'baggingRF'

# Created by hjin on 5/19/2016
"""
This application is to do bagging RF without CV and grid search in ML module
for multi-simulation
Also use Spark-csv package to read in CSV file.

please change line 43 to line 51 to your own variables

Application has an argument, it is the app name
"""

import sys
import os
import time
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.mllib.linalg import Vectors
import numpy as np
#import pandas as pd
import random
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#constants variables
app_name = sys.argv[1]
path = "s3://emr-rwes-pa-spark-dev-datastore/Hui/shire_test"
data_path = path + "/01_data/data_973"
pos_file = "/dat_hae.csv"
neg_file ="/dat_nonhae.csv"

#change before running
ts_prop = 0.2
num_simu = 5
ratio = 50

numTrees = 30
numDepth = 5

par = 400
seed = 42

random.seed(seed)
seed_seq = [random.randint(10, 100) for i in range(num_simu)]

nIter = int(200/ratio)
start_time = time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d')
resultDir_s3 = path + "/02_result/" + app_name + st + "/"

#define the functions will be used
#function 1: split the column in RDD into different columns in dataframe
def Parse(pair):
    return dict(list(pair[0].asDict().items()) + [("iterid", pair[1])])

#function 2: get the predicted probability in Vector
def getitem(i):
    def getitem_(v):
         return v.array.item(i)
    return udf(getitem_, DoubleType())

#function 3: register the DataFrame as table
def regit(data, iterid):
    return data[iterid].registerTempTable(('ls_' + str(iterid)))

#function 4: create SQL query
def sqlqu(nIter):
    sqla = 'SELECT tb0.patid AS patid, tb0.label AS label, (tb0.prob_1'
    for i in range(1, nIter):
        sqla = sqla + '+tb' + str(i) + '.prob_1'
    sqlb = ')/' + str(nIter) + ' AS avg_prob FROM ls_0 AS tb0'
    for j in range(1, nIter):
        sqlb = sqlb + ' INNER JOIN ls_' + str(j) + ' AS tb' + str(j)\
               + ' ON tb0.patid = tb' + str(j) +'.patid'
    sqlquery = sqla + sqlb
    return sqlquery

#function 5: bagging random forest
def baggingRF(iterid, neg_tr_iterid, pos_tr, ts):

    #select the Non-HAE patient by iteration ID
    ineg_tr = neg_tr_iterid\
        .filter(neg_tr_iterid.iterid == iterid)\
        .select('patid', 'label', 'features')

    #combine with positive training data
    itr = pos_tr.unionAll(ineg_tr)

    #create the labelIndexer
    #transfer to RF invalid label column
    labelIndexer = StringIndexer(inputCol="label",outputCol="indexedLabel").fit(itr)

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel",
                                maxDepth=numDepth, numTrees=numTrees, seed=seed)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, rf])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(itr)

    # Make predictions.
    predictions = model.transform(ts)
    pred_score_ts = predictions.select(
            predictions.patid, predictions.label,
            getitem(0)('probability').alias('prob_0'),
            getitem(1)('probability').alias('prob_1'))

    #evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
    # predictionCol="prediction",
     #metricName="precision")
    #ppv = evaluator.evaluate(predictions)
    return pred_score_ts


#function 6: main function
def main(sc, isim, pos_ori, neg_ori):

    #random split in positive cases and use HAE patient ID to select Non-HAE
    (pos_tr, pos_ts) = pos_ori.randomSplit([(1-ts_prop), ts_prop],
                                           seed=seed_seq[isim])

    #using HAE patient ID to select Non-HAE patient and only keep Non-HAE
    neg_tr = neg_ori\
        .join(pos_tr, neg_ori.hae_patid == pos_tr.patid,'inner')\
        .select(neg_ori.patid, neg_ori.label, neg_ori.hae_patid,neg_ori.features)

    #!!!after the merging, the Non-HAE patients ordered by hae_patid!!!
    #create iteration ID in Non-HAE
    nData = pos_tr.count()*200
    npIterIDs = np.array(list(range(nIter))*np.ceil(float(nData)/nIter))
    rddIterIDs = sc.parallelize(npIterIDs).map(int)

    #add index for Non-HAE patients
    neg_tr_Indexed = neg_tr\
        .rdd\
        .zipWithIndex()\
        .toDF()\
        .withColumnRenamed("_1", "orgData")

    #add index for iteration ID
    dfIterIDs = rddIterIDs\
        .zipWithIndex()\
        .toDF()\
        .withColumnRenamed("_1", "iterid")

    #merge them together to get the Non-HAE patients with iteration ID
    neg_tr_iterid = neg_tr_Indexed\
        .join(dfIterIDs, "_2")\
        .drop('_2')\
        .map(Parse)\
        .coalesce(par)\
        .toDF()\
        .drop('hae_patid')\
        .cache()

    #test set is the rows in original negative cases but not in training set
    neg_ts = neg_ori.subtract(neg_tr).drop('hae_patid')

    #combine to test data
    ts = pos_ts.unionAll(neg_ts)

    #do loops on baggingRF function
    pred_ts_ls = [baggingRF(iterid, neg_tr_iterid, pos_tr, ts) for iterid in
                  range(nIter)]

    return pred_ts_ls


if __name__ == "__main__":

    sc = SparkContext(appName = app_name)

    sqlContext = SQLContext(sc)

    #reading in the data from S3
    pos = sqlContext.read.load((data_path + pos_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true').repartition(par)

    neg = sqlContext.read.load((data_path + neg_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true').repartition(par)
    #see the column names
    pos_col = pos.columns
    neg_col = neg.columns

    #combine features
    assembler_pos = VectorAssembler(inputCols=pos_col[2:],outputCol="features")
    assembler_neg = VectorAssembler(inputCols=neg_col[2:-1],outputCol="features")

    #get the input positive and negative dataframe
    pos_asmbl = assembler_pos.transform(pos)\
        .select('PATIENT_ID', 'HAE','features')\
        .withColumnRenamed('PATIENT_ID', 'patid')\
        .withColumnRenamed('HAE', 'label')

    pos_ori = pos_asmbl.withColumn('label', pos_asmbl['label'].cast('double'))

    neg_asmbl = assembler_neg.transform(neg)\
        .select('PATIENT_ID', 'HAE', 'HAE_PATIENT_ID', 'features')\
        .withColumnRenamed('PATIENT_ID', 'patid')\
        .withColumnRenamed('HAE', 'label')\
        .withColumnRenamed('HAE_PATIENT_ID', 'hae_patid')

    neg_ori = neg_asmbl.withColumn('label', neg_asmbl['label'].cast('double'))

    #doing the main function in for loop as well, return as list
    loop_re = [main(sc, isim=isim, pos_ori=pos_ori, neg_ori=neg_ori ) for
               isim in range(num_simu)]

    #averaging the prediting scores in each loop
    for isim in range(num_simu):

        #extract results in each loop
        result_ls = loop_re[isim]

        #register DataFrame as a table
        for iterid in range(nIter):
            regit(result_ls, iterid)

        #create the SQL queries
        sql_query = sqlqu(nIter)

        #using SQL queries to create dataframe including mean of prob
        avg_pred_ts = sqlContext.sql(sql_query)

        #output the predicted scores to S3
        avg_pred_ts.write.format("com.databricks.spark.csv")\
            .save((resultDir_s3 +"avg_pred_ts" + isim + ".csv"),header="true")

    sc.stop()








