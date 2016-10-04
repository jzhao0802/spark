__author__ = 'hjin'
__project__ = 'baggingRF'

# Created by hjin on 5/19/2016
"""
This application is to do bagging RF without CV and grid search in ML module
for multi-simulation
Also use Spark-csv package to read in CSV file.

please change line 45 to line 81 to your own variables

Application has an argument, it is the app name
"""

import sys
import os
import time
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
import numpy as np
#import pandas as pd
import random
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#from test_crossvalidator import CrossValidatorWithStratification
#constants variables
app_name = sys.argv[1]
s3_path = "s3://emr-rwes-pa-spark-dev-datastore"
par = 2
seed = 42
ori_ratio = 200

####!!!!!change before running!!!######################################
#path
data_path = s3_path + "/BI_IPF_2016/01_data/"
s3_outpath = s3_path + "/lichao.test/Results/"
master_path = "/home/lichao.wang/code/lichao/test/Results/"

# data file
pos_file = "ipf_sample.csv"
neg_file ="nonipf_ac_sample.csv"

#test proportion
ts_prop = 0.2

#number of simulation
num_sim = 2

#target ratio in each bagging RF model
target_ratio = 100

#Grid-search and Cross Validation flag
gscv_flag = False

#!!!!!!!!!if gscv_flag=True, change here!!!!!
numtree = [200, 300]
numdepth = [3, 4]
nodesize = [3, 5] # around 5, integer
mtry = ['onethird', 'sqrt'] #'auto', 'all, 'onethird', 'sqrt', 'log2'
fold = 5
CVCriteria = 'areaUnderPR' #areaUnderROC or areaUnderPR

#!!!!!!!!!!!if gscv_flag=False, change here!!!!!
num_tree = 10
num_depth = 2
node_size = 5 # around 5, integer
m_try = 'auto' #'auto', 'all, 'onethird', 'sqrt', 'log2'

#######!!!Setting End!!!!#######################################

#number of iteration
nIter = int(ori_ratio/target_ratio)

#seed
random.seed(seed)
seed_seq = [random.randint(10, 100) for i in range(num_sim)]

#S3 output folder
start_time = time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
resultDir_s3 = s3_outpath + app_name + '_' + st + "/"

#master node output folder
resultDir_master = master_path + app_name + '_' + st + "/"
if not os.path.exists(resultDir_master):
    os.makedirs(resultDir_master)
os.chmod(resultDir_master, 0o777)

#define the functions will be used
#function to add simulation or iteration ID
# Attention! This function is not only adding an ID column!
def addID(dataset, number, npar, name):
    nPoses = dataset.count()
    npFoldIDsPos = np.array(list(range(number)) * np.ceil(float(nPoses) / number))
    # select the actual numbers of FoldIds matching the count of positive data points
    npFoldIDs = npFoldIDsPos[:nPoses]
    # Shuffle the foldIDs to give randomness
    np.random.shuffle(npFoldIDs)
    rddFoldIDs = sc.parallelize(npFoldIDs, npar).map(int)
    dfDataWithIndex = dataset.rdd.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "orgData")
    dfNewKeyWithIndex = rddFoldIDs.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "key")
    dfJoined = dfDataWithIndex.join(dfNewKeyWithIndex, "_2") \
        .select("orgData.matched_positive_id", 'key') \
        .withColumnRenamed('key', name) \
        .coalesce(npar)
    return dfJoined

#function to assembel average prob_1 and average_prob_0 to vector dense
def avg_asmb(data):
    newdata = data\
        .withColumn('avg_prob_0',lit(1)-data.avg_prob)\
        .withColumnRenamed('avg_prob', 'avg_prob_1')
    asmbl = VectorAssembler(inputCols=['avg_prob_0', 'avg_prob_1'],
                           outputCol="raw")
    #get the input positive and negative dataframe
    data_asmbl = asmbl.transform(newdata)\
        .select('patid', 'label', 'raw')
    return data_asmbl

#function 1: get the predicted probability in Vector
def getitem(i):
    def getitem_(v):
         return v.array.item(i)
    return udf(getitem_, DoubleType())

#function 2: generate excluded variable list
#def exc_list(d_path, file):
#    data = np.loadtxt(d_path + file ,dtype=np.str,delimiter=',',skiprows=0)
#    var_ls = data[1:, 0].tolist()
#    var_flag_ls = [x + '_FLAG' for x in var_ls]
#    var_avg_ls = [x + '_AVG_RXDX' for x in var_ls]
#    var_exc_ls = var_flag_ls + var_avg_ls
#    return var_exc_ls

#function 3: register the DataFrame as table
#def regit(data, iterid, name):
#    return data[iterid].registerTempTable((name + str(iterid)))

#function 4: create SQL query to calcualte average probability
#def sqlqu1(nIter):
#    sqla = 'SELECT tb0.patid AS patid, tb0.label AS label, (tb0.prob_1'
#    for i in range(1, nIter):
#        sqla = sqla + '+tb' + str(i) + '.prob_1'
#    sqlb = ')/' + str(nIter) + ' AS avg_prob FROM ls_0 AS tb0'
#    for j in range(1, nIter):
#        sqlb = sqlb + ' INNER JOIN ls_' + str(j) + ' AS tb' + str(j)\
#               + ' ON tb0.patid = tb' + str(j) +'.patid'
#    sqlquery = sqla + sqlb
#    return sqlquery

#function 5: create SQL query to union all prob across simulation
#def sqlqu2(nsim):
#    iquery = 'SELECT * FROM sim_0 UNION ALL '
#    for i in range(1, nsim-1):
#        iquery = iquery + 'SELECT * FROM sim_' + str(i) + ' UNION ALL '
#    query = iquery + 'SELECT * FROM sim_' + str(nsim-1)
#    return query


#function 6: decide which column of probability we would like to select
#we create this function because sometimes the StringIndexer would revert label
#we are intested in probability of 1
def getprob(pred):
    #get the first observation
    #???? need improve???
    firstone = pred.take(1)
    firstonezip = zip(*firstone)
    #get the true label and indexed label from prediction
    true_label = firstonezip[1]
    index_label = firstonezip[2]
    #compare true label and index label, if not equal then revert
    if (true_label != index_label):
        pred_revert = pred\
            .withColumnRenamed('prob_0','temp1')\
            .withColumnRenamed('prob_1','prob_0')\
            .withColumnRenamed('temp1','prob_1')
    else:
        pred_revert = pred
    return pred_revert

#function 7: PR curve calculation
def pr_curve(data, prob,resultDir_s3, output):
    #get label and prob from the dataframe
    temp = data.withColumnRenamed(prob, 'prob_1')
    labelsAndProbs =temp\
        .withColumn('prob_1', round(temp.prob_1,3))\
        .select(["label", 'prob_1'])
    #get distinct threshold values
    thresholds_interim = labelsAndProbs\
        .select(col('prob_1').alias("threshold"))\
        .distinct()
    num_partitions = data.rdd.getNumPartitions()
    thresholds = thresholds_interim.coalesce(num_partitions)
    #cache dataframes
    labelsAndProbs.cache()
    thresholds.cache()
    cartProduct = thresholds\
        .rdd\
        .cartesian(labelsAndProbs.rdd)\
        .toDF()\
        .withColumn("threshold",col("_1").threshold)\
        .withColumn("label", col("_2").label)\
        .withColumn("prob_1", col("_2").prob_1)\
        .drop("_1")\
        .drop("_2")
    PRs = cartProduct\
    	.withColumn("pred", when(col('prob_1') > col("threshold"),1.0).otherwise(0.0))\
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
        .coalesce(1).collect()
        # .coalesce(1)\
        # .write.save((resultDir_s3+output),"com.databricks.spark.csv",header="true")

#function 8: bagging random forest
def baggingRF(iiter, isim, tr_neg_simdata, tr_pos_simdata, tr_iterid,
              val_simdata, gscv_flag=gscv_flag, numtree=numtree,
              numdepth=numtree, nodesize=nodesize, mtry=mtry,
              num_depth=num_depth, num_tree=num_tree, node_size=node_size,
              m_try=m_try, seed=seed, CVCriteria=CVCriteria):
    #select negative patients for each iteration
    pat_iterid = tr_iterid.filter(tr_iterid.iterid == iiter)
    #select corresponding negative patients
    neg_iterdata = tr_neg_simdata\
        .join(pat_iterid, pat_iterid["matched_positive_id"] ==tr_neg_simdata["matched_positive_id"], 'inner')\
        .select(tr_neg_simdata["patid"], tr_neg_simdata["label"],
                tr_neg_simdata["matched_positive_id"], tr_neg_simdata["features"])
    #combine with positive training data
    #????do we need to repartition or coalesce here???
    itr = tr_pos_simdata.unionAll(neg_iterdata)
    #create the labelIndexer
    #transfer to RF invalid label column
    #labelIndexer = StringIndexer(inputCol="label",outputCol="indexedLabel").fit(itr)
    if gscv_flag == True:
        #rename patid to matched_positive_id
        # itr = itr.withColumnRenamed('patid', 'matched_positive_id')
        labelIndexer = StringIndexer(inputCol="label",outputCol="indexedLabel").fit(itr)
        # Train a RandomForest model.
        rf = RandomForestClassifier(labelCol="indexedLabel", seed=seed)
        # Chain indexers and forest in a Pipeline
        pipeline = Pipeline(stages=[labelIndexer, rf])
        paramGrid = ParamGridBuilder()\
            .addGrid(rf.numTrees, numtree)\
            .addGrid(rf.maxDepth, numdepth)\
            .addGrid(rf.minInstancesPerNode, nodesize)\
            .addGrid(rf.featureSubsetStrategy, mtry)\
            .build()
        # Create the evaluator
        evaluator = BinaryClassificationEvaluator(metricName=CVCriteria)
        #Create the cross validator
        #using datafactz's crossvalidator
        crossval = CrossValidatorWithStratification(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=fold)
        # Train model.  This also runs the indexers.
        model = crossval.fit(itr)
    
        #????store the best parameters
        best_parameter = model.getBestModelParms()
        best_prm_file = open(resultDir_master + 'best_prm_sim' + str(isim) +
                            '_iter' + str(iiter) +'.txt', "w")
        best_prm_file.writelines(best_parameter)
        best_prm_file.close()
        os.chmod(resultDir_master + 'best_prm_sim' + str(isim) +
                            '_iter' + str(iiter) +'.txt', 0o777)
    
        # Make predictions.
        predictions = model.transform(val_simdata)
    
        #pred_scoring_sample = model.transform(scoring_sampling)
        pred_score_val = predictions.select(predictions["patid"],
                                            predictions["matched_positive_id"],
                                            predictions["label"],
                                            predictions["indexedLabel"],
                                            getitem(0)('probability').alias('prob_0'),
                                            getitem(1)('probability').alias('prob_1'))
    elif gscv_flag == False:
        labelIndexer = StringIndexer(inputCol="label",outputCol="indexedLabel").fit(itr)
        # Train a RandomForest model.
        rf = RandomForestClassifier(labelCol="indexedLabel",
                                    maxDepth=num_depth, numTrees=num_tree,
                                    minInstancesPerNode=node_size,
                                    featureSubsetStrategy= m_try, seed=seed)
        # Chain indexers and forest in a Pipeline
        pipeline = Pipeline(stages=[labelIndexer, rf])
        # Train model.  This also runs the indexers.
        model = pipeline.fit(itr)
        # Make predictions.
        predictions = model.transform(val_simdata)
        #pred_scoring_sample = model.transform(scoring_sampling)
        pred_score_val = predictions.select(predictions["patid"],
                                            predictions["matched_positive_id"],
                                            predictions["label"],
                                            predictions["indexedLabel"],
                                            getitem(0)('probability').alias('prob_0'),
                                            getitem(1)('probability').alias('prob_1'))
    pred_score_val2 = getprob(pred_score_val)
    #evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
    # predictionCol="prediction",
     #metricName="precision")
    #ppv = evaluator.evaluate(predictions)
    return pred_score_val2


#function 8: simulation function
def simulation(isim, patsim, pos_ori, neg_ori):

    #allocate patients into training / validation
    val_simid = patsim.filter(patsim.simid == isim)
    tr_simid = patsim.filter(col("simid") != isim)

    #join with original positive patients and negative patients
    #????not sure whether use neg_ori.join(valsimid) or the other way around????
    #???? should we Broadcast the valsimid dataframe since it is quite small????
    val_neg_simdata = neg_ori\
        .join(val_simid, val_simid["matched_positive_id"] == neg_ori["matched_positive_id"],'inner')\
        .select(neg_ori["patid"], neg_ori["label"], neg_ori["matched_positive_id"],
                neg_ori["features"])
    tr_neg_simdata = neg_ori.filter(col("matched_positive_id") != isim)

    val_pos_simdata = pos_ori\
        .join(val_simid, tr_simid["matched_positive_id"] == pos_ori["matched_positive_id"],'inner')\
        .select(pos_ori["patid"], pos_ori["label"], pos_ori["matched_positive_id"],pos_ori["features"])

    tr_pos_simdata = pos_ori.filter(col("patid") != isim)

    #combine the validation data
    val_simdata = val_pos_simdata.unionAll(val_neg_simdata)

    #create a new column in tr_simid dataframe to identify the iteration
    tr_patid_pos = tr_simid.select("matched_positive_id")
    tr_iterid = addID(tr_patid_pos, nIter, par, 'iterid')

    # cache reusable dataframes
    tr_neg_simdata.cache()
    tr_pos_simdata.cache()
    tr_iterid.cache()
    val_simdata.cache()

    # run bagging RF model on iterations return the prediction on validation data
    # return a list, each element is a dataframe of predcited scores on validation
    pred_val_bRF_ls = [baggingRF(iiter, isim, tr_neg_simdata, tr_pos_simdata,
                             tr_iterid, val_simdata) for iiter in range(nIter)]

    # union all dataframes in pred_val_bRF_ls
    pred_val_bRF = reduce(DataFrame.unionAll, pred_val_bRF_ls)

    #cache the dataframe
    pred_val_bRF.cache()

    #geting average of prob_1 by group by
    avg_pred_val = pred_val_bRF.groupBy('patid','label')\
        .agg(avg(col('prob_1')).alias('avg_prob'))\
        .coalesce(pred_val_bRF_ls[0].rdd.getNumPartitions())

    # cache the dataframe
    avg_pred_val.cache()

    #output the predicted scores to S3
    avg_pred_val.coalesce(1).write.save((resultDir_s3+"avg_pred_val_sim" + str(isim) + ".csv"),
                      "com.databricks.spark.csv",header="true")

    # AUC & AUPR
    # Create the evaluator
    evaluator2 = BinaryClassificationEvaluator(rawPredictionCol="raw")
    avg_pred_val_v = avg_asmb(avg_pred_val)
    sim_AUC_val = evaluator2.evaluate(avg_pred_val_v,
                                 {evaluator2.metricName:'areaUnderROC'})
    sim_AUPR_val = evaluator2.evaluate(avg_pred_val_v,
                                 {evaluator2.metricName:'areaUnderPR'})


    #print out AUC results
    sim_auc = "average validation data AUC = %s " % sim_AUC_val + "\n"
    sim_aupr = "average validation data AUPR = %s " % sim_AUPR_val + "\n"
    sim_auc_file = open(resultDir_master + 'AUC_AUPR_sim' + str(isim) + '.txt',
                     "w")
    sim_auc_file.writelines(sim_auc)
    sim_auc_file.writelines(sim_aupr)
    sim_auc_file.close()
    os.chmod(resultDir_master + 'AUC_AUPR_sim' + str(isim) + '.txt', 0o777)

    return avg_pred_val

#function 9: main function
def main(sc, data_path=data_path, pos_file=pos_file, neg_file=neg_file):

    #reading in the data from S3
    pos = sqlContext.read.load((data_path + pos_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')

    neg = sqlContext.read.load((data_path + neg_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')

    # change the columns: 
    # both pos and neg have column patid, which is the actual patienet id
    # they both also have column matched_positive_id, which is for matching the pos and neg
    # For pos, column matched_positive_id has the same values as column patid
    pos = pos.drop("nonipf_patid")
    pos = pos.withColumn("matched_positive_id", pos["patid"])
    neg = neg.withColumnRenamed("patid", "matched_positive_id")\
             .withColumnRenamed("nonipf_patid", "patid")

    #reading in excluded variable list from master node
    #exc_var_list = exc_list(master_data_path, exc_file)

    #see the column names
    pos_col = pos.columns

    # feature list
    common_list = ['patid', 'label', 'matched_positive_id']
    inc_var = [x for x in pos_col if x not in common_list]
    #exc_var_list+

    #combine features
    assembler = VectorAssembler(inputCols=inc_var,outputCol="features")

    #get the input positive and negative dataframe
    pos_asmbl = assembler.transform(pos)\
        .select('patid', 'label', 'matched_positive_id', 'features')

    pos_ori = pos_asmbl.withColumn('label', pos_asmbl['label'].cast('double'))

    neg_asmbl = assembler.transform(neg)\
        .select('patid', 'label', 'matched_positive_id', 'features')

    neg_ori = neg_asmbl.withColumn('label', neg_asmbl['label'].cast('double'))

    # union All positive and negative data as dataset
    # dataset = pos_ori.unionAll(neg_ori)

    #create a dataframe which has 2 column, 1 is patient ID, other one is simid
    patid_pos = pos_ori.select('matched_positive_id')
    patsim = addID(patid_pos, num_sim, par, 'simid')

    #cache reusable dataframes
    patsim.cache()
    pos_ori.cache()
    neg_ori.cache()

    # run simulation funciton on num_sim, output the average predicted score to S3
    avg_pred_val_ls = [simulation(isim, patsim, pos_ori, neg_ori) for isim in \
            range(num_sim)]

    # union all dataframes in avg_pred_val_ls
    avg_pred_all = reduce(DataFrame.unionAll, avg_pred_val_ls)

    #cache the dataframe
    avg_pred_all.cache()

    #output the predicted scores to S3
    avg_pred_all.write.save((resultDir_s3+"avg_pred_all.csv"),
                      "com.databricks.spark.csv",header="true")

    # AUC & AUPR
    # Create the evaluator
    evaluator3 = BinaryClassificationEvaluator(rawPredictionCol="raw")
    avg_pred_all_v = avg_asmb(avg_pred_all)
    AUC_all = evaluator3.evaluate(avg_pred_all_v,
                                 {evaluator3.metricName:'areaUnderROC'})

    AUPR_all = evaluator3.evaluate(avg_pred_all_v,
                                 {evaluator3.metricName:'areaUnderPR'})

    #print out AUC results
    auc_all = "Training data AUC = %s " % AUC_all + "\n"
    aupr_all = "Training data AUPR = %s " % AUPR_all + "\n"
    auc_all_file = open(resultDir_master + 'AUC_AUPR_all.txt', "w")
    auc_all_file.writelines(auc_all)
    auc_all_file.writelines(aupr_all)
    auc_all_file.close()
    os.chmod(resultDir_master + 'AUC_AUPR_all.txt', 0o777)

    #output PR curve for dataset through simulation
    pr_curve(avg_pred_all, 'avg_prob', resultDir_s3=resultDir_s3,
             output='PR_curve_alldata/')


if __name__ == "__main__":

    sc = SparkContext(appName = app_name)

    sqlContext = SQLContext(sc)

    main(sc)

    sc.stop()
    
