# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:11:03 2016

Modified by Aditya (DatFactZ, Inc.) on Mon Jul 11 5:19:04 2016

@author: zywang
This function is used to calculate the precision, recall and FPR. 
    1. precision and recall is used for PR curve.
    2. recall and FPR is used for ROC curve.

The following are some parameters:
    datapath:     the overall folder path
    inpath:       the path of data
    filename:     file used to calculate precision, recall and fpr. 
    posProb_name: the name of positive probability
    response_name:the name of origin true label
    par:          number of partitions
    output:       output path
"""

import sys
import os
import time
import datetime
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import *

def PrecisionRecall_calculation(datapath, inpath, filename, response_name,\
                                posProb_name, output):  
    
    #read data from csv using package com.databricks.spark.csv
    data = sqlContext.read.load(datapath + inpath + filename, 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')\
        .withColumnRenamed(response_name,'label')\
        .withColumnRenamed(posProb_name,'prob_1')
    
    #get label and prob from the dataframe                
    labelsAndProbs = data.withColumn('prob', round(data.prob_1,3)).select(["label", "prob"])

    #get distinct threshold values
    thresholds_interim = labelsAndProbs.select(col('prob').alias("threshold")).distinct()
    
    """Getting the number of partitions for thresholds dataframe
    to perform coalesce (to reduce the number of partitions).
    In other words to reduce number of tasks in spark job stage"""
    num_partitions = thresholds_interim.rdd.getNumPartitions()
    thresholds = thresholds_interim.coalesce(num_partitions/2)
    
    #cache dataframes
    labelsAndProbs.cache()
    thresholds.cache() 
    
    """Cross join thresholds and labelsAndProbs dataframes by '.join' operation.
    Create column 'pred' which has 1.0 when prob is greater than threshold, else 0.0.
    Create column bTP (True positive), bFP (False Positive), bTN (True Negative), bFN (False Positive)
    based on the column 'pred' and 'label' using '.withColumn', '.when' and '.otherwise' dataframe DSL columnar methods.
    
    Group by 'threshold' 
    aggregate the dataframe after above group by and sum 'bTP' and name the column as 'nTPs' using .alias
    sum 'bFP' and name the column as 'nFPs' using .alias
    sum 'bTN' and name the column as 'nTNs' using .alias
    sum 'bFN' and name the column as 'nFNs' using .alias
    
    Inside the final select statement,
    Precision is calculated based on the formulae TP/(TP+FP+1e-9), and round to 3 decimal places,
    Recall is calculated based on the formulae TP/(TP+FN+1e-9), and round to 3 decimal places,
    theshold is selected as it is from previous step 
    and call .coalesce(1) to save in a single partition
    
    Finally save the dataframe(with 3 columns i.e. threshold, precision and recall) as csv usins '.save' method."""
    PRs = thresholds.join(labelsAndProbs)\
    	.withColumn("pred", when(col("prob") > col("threshold"),1.0).otherwise(0.0))\
    	.withColumn("bTP", when((col("label") == col("pred")) & (col("pred") == 1),1.0).otherwise(0.0))\
    	.withColumn("bFP", when((col("label") != col("pred")) & (col("pred") == 1),1.0).otherwise(0.0))\
    	.withColumn("bTN", when((col("label") == col("pred")) & (col("pred") == 0),1.0).otherwise(0.0))\
    	.withColumn("bFN", when((col("label") != col("pred")) & (col("pred") == 0),1.0).otherwise(0.0))\
        .select(col("threshold"),col("bTP"),col("bFP"),col("bTN"),col("bFN"))\
                    .groupBy("threshold")\
                    .agg(sum(col("bTP")).alias("nTPs"), 
                         sum(col("bFP")).alias("nFPs"),
                         sum(col("bTN")).alias("nTNs"), 
                         sum(col("bFN")).alias("nFNs"))\
        .select(round(col("nTPs") / (col("nTPs") + col("nFPs") + 1e-9),3).alias("precision"),
			round(col("nTPs") / (col("nTPs") + col("nFNs") + 1e-9),3).alias("recall"),
			col("threshold"))\
        .coalesce(1)\
        .save(output,"com.databricks.spark.csv",header="true")


if __name__ == "__main__":

    # Some constance change here!!!!
    datapath = "hdfs://ec2-52-90-17-145.compute-1.amazonaws.com:9000/data"
    #"s3://emr-rwes-pa-spark-dev-datastore/Hui/shire_test/02_result/"

    inpath = "/"

    #"lasso_db4_20160627_065857/"
    posProb_name = "Prob_1"
    response_name = "label"

    #set the command line parameters
    flag = sys.argv[1]
    app_name = sys.argv[2]

    #filename = 'pred_ts_sim0.csv/part-00000'
    filename = 'pred_score_ts'
    
    #create SparkConf
    conf = SparkConf()
    conf.setAppName(app_name)
   
    '''conset("spark.dynamicAllocation.enabled", "true")
    conset("spark.shuffle.service.enabled", "true")
    conset("spark.dynamicAllocation.maxExecutors", "30")'''

    #create SparkContext
    sc = SparkContext(conf = conf)
    
    #create SQLContext
    sqlContext = SQLContext(sc)
    
    #Create the output folder
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    output = datapath + 'PR_curve_' + st + "/"
    if not os.path.exists(output):
        os.makedirs(output)

    #Calculate the precision, recall and FPR according to unique positive probability
    PrecisionRecall_calculation(datapath, inpath, filename, response_name, \
                                posProb_name, output)
    
