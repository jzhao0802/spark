# -*- coding: utf-8 -*-
'''
Created on Tue Jun  7 13:51:37 2016
@author: zywang

This is used to do the descriptive statistic and cross tables 
between response and features

'''
import os
import time
import datetime
from pyspark import SparkContext
from pyspark.sql import SQLContext
import numpy as np

# Some parameters
data_path = 's3://emr-rwes-pa-spark-dev-datastore/'
inpath = 'Shire_test/200k_patid/'
datapath = data_path + inpath
app_name = "Descriptive_Statistic"
filename = 'tr_patid.csv'
response = 'HAE'
minrate = float(0.02)
par = int(300)

def data_ds(sc, datapath, outpath, filename, par):
    from pyspark.mllib.stat import Statistics
    #Reading in data as RDD data
    data1 = sc.textFile(datapath + filename, par) #in RDD format
    #Skip the header, and re-structure the columns as "label" and "features"
    header = data1.first()
    data2 = data1.filter(lambda x: x != header).map(lambda line: line.split(','))
    # split all header
    head = header.split(",")
    #Descriptive Statistics
    cStats = Statistics.colStats(data2)
    avg = cStats.mean()
    SD = cStats.variance()
    cntNon0 = cStats.numNonzeros()
    maxValue = cStats.max()
    minValue = cStats.min()
    ds = np.vstack((head, cntNon0, minValue, avg, SD, maxValue)).T.tolist()
    rds = sc.parallelize(ds, 1)
    rds2 = sqlContext.createDataFrame(rds, ["Variable","cntNonZero","Min","Avg","SD","Max"])
    rds2.registerTempTable("rds2")
    sql_query = 'select * from rds2'
    sqlContext.sql(sql_query)\
    .save(outpath + "Descriptive_Tb.csv", "com.databricks.spark.csv",header="true")
   
def cross_tb(sc, datapath, outpath, filename, par, response, minrate):
    from pyspark.sql.functions import lit
        
    #reading in data as RDD data
    data1 = sqlContext.read.load(datapath + filename, 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')
    head = data1.columns
    
    # Select the features less than 3 levels and obtain the table
    def unique_cnt(data, colnames, i):
        temp = data.select(colnames[i]).distinct().count()
        if (temp < 3):
            return(colnames[i])
    
    fe = [unique_cnt(data1, head, i) for i in range(len(head))]
    feature_select = [x for x in fe if x is not None]  
    data_df = data1.select(feature_select)   
    feature = [x for x in data_df.columns if x not in response]
    
    #Calculate the cross table between features and response
    cs_tb = data_df.stat.crosstab(feature[0], response)\
            .withColumnRenamed(feature[0] + '_' + response,'value')\
            .withColumnRenamed('0', 'response0')\
            .withColumnRenamed('1', 'response1')\
            .withColumn("Variable", lit(feature[0]))    
    for j in range(1, len(feature)):
        tb = data_df.stat.crosstab(feature[j], response)\
               .withColumnRenamed(feature[j] + '_' + response,'value')\
               .withColumnRenamed('0', 'response0')\
               .withColumnRenamed('1', 'response1')\
               .withColumn("Variable", lit(feature[j]))
        cs_tb = cs_tb.unionAll(tb)
    
    # Rename the columns and Output the cross table
    cs_tb1 = cs_tb.coalesce(1)
    cs_tb1.registerTempTable("cs_tb1")
    sql_query = 'select Variable, value, response0, response1 from cs_tb1'    
    cs_tb2 = sqlContext.sql(sql_query)
        
    d = data1.select(response).withColumnRenamed(response,'response')
    d1 = d.select(d.response.cast("int").alias('response'))
    tcnt = d1.groupBy().sum("response").collect()[0][0]
    cs_tb2 = cs_tb2.withColumn('response_rate', cs_tb2.response1/tcnt)
    cs_tb2.save(outpath + "Cross_Table.csv", "com.databricks.spark.csv",\
                header="true")
    
    #Select the variables with positive response rate less than minrate
    query2 = 'select Variable, response_rate,'\
             + 'case when response_rate in (0, 1) then "Completed separated" '\
             + 'else "Less than minimun rate" end as Comments ' \
             + 'from cs_tb2_t where '\
             + 'value = "1" and (response_rate <= ' \
             + str(minrate) + ' or response_rate >= ' + str(1-minrate) + ')'
    cs_tb2.registerTempTable("cs_tb2_t")
    cs_tb3 = sqlContext.sql(query2)
    cs_tb3.save(outpath + "Dropped_variables.csv", "com.databricks.spark.csv",\
                header="true")

if __name__ == "__main__":
    sc = SparkContext(appName = app_name)
    sqlContext = SQLContext(sc)
    
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')  
    resultDir_s3 = data_path + "zy_test/" + st + "/"
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3)
    
    data_ds(sc = sc, datapath = datapath, outpath = resultDir_s3, 
            filename = filename, par = par)
    cross_tb(sc = sc, datapath = datapath, outpath = resultDir_s3, 
             filename = filename, par = par, response = response, 
             minrate = minrate)

