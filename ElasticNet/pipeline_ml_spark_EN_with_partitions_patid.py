'''
This function could run LR Elastic-Net with Spark-submit in Spark ML module.
elasticNetParam corresponds to alpha and regParam corresponds to lambda
1. elasticNetParam is in range [0, 1]. 
    For alpha = 0, the penalty is an L2 penalty. 
    For alpha = 1, it is an L1 penalty.
2. regParam is regularization parameter (>= 0).
'''

# Import the packages
import sys
import os
import time
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
import numpy as np

#from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Some constance
data_path = 's3://emr-rwes-pa-spark-dev-datastore/'
master_path = "/home/zwang/test_codes/Results/"
tr_file = 'data_forModel.csv'
ts_file = 'data_forModel.csv'
app_name = 'LR_Hyper_CV_simple'
inpath = 'zy_test/model_data/'
# For lambda
lambdastart = float(0.1)
lambdastop = float(1)
lambdanum = int(10)
# For alpha
alphastart = float(1)
alphastop = float(1)
alphanum = int(2)
# For cross validation
fold = int(2)
par = int(300)
CVCriteria = 'areaUnderROC' #areaUnderROC or areaUnderPR

def parameter_ck():
    global alphanum, lambdanum
    errorCnt = 0
    if (alphastart < 0 or alphastop > 1 or lambdastart < 0):
        errorCnt += 1
    if (alphastart == alphastop):
        alphanum = int(1)
    if (lambdastart == lambdastop):
        lambdanum = int(1)
    return(errorCnt)

# Define the functions
def load_csv_file(sc, data_path, inpath, file, par):
    global header
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
    
def getitem(i):
    def getitem_(v):
         return v.array.item(i)
    return udf(getitem_, DoubleType())
    
def main(sc, app_name=app_name, 
         data_path=data_path, inpath=inpath, master_path=master_path,
         tr_file=tr_file, ts_file=ts_file, 
         lambdastart=lambdastart, lambdastop=lambdastop, lambdanum=lambdanum, 
         alphastart=alphastart,alphastop=alphastop, alphanum=alphanum,
         CVCriteria=CVCriteria,fold=fold, par=par):
             
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = data_path + "zy_test/" + st + "/"
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3, 0777)
        
    resultDir_master = master_path + st + "/"
    if not os.path.exists(resultDir_master):
        os.makedirs(resultDir_master, 0777)
        
    #reading in data
    tr = load_csv_file(sc, data_path, inpath, tr_file, par)
    ts = load_csv_file(sc, data_path, inpath, ts_file, par)
    # Build the model
    lr = LogisticRegression(featuresCol = "features", 
                            labelCol = "label",
                            fitIntercept = True)                         
                            #,standardization = False) This isn't usable in spark 1.5.2 but could be used in spark 1.6.1
    # Create the pipeline
    # pipeline = Pipeline(stages=[lr])
    # Create the parameter grid builder
    paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, list(np.linspace(lambdastart, lambdastop, lambdanum)))\
    .addGrid(lr.elasticNetParam, list(np.linspace(alphastart, alphastop, alphanum)))\
    .build()
    # Create the evaluator
    evaluator = BinaryClassificationEvaluator(metricName=CVCriteria)
    #Create the cross validator
    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=fold)
    #run cross-validation and choose the best parameters
    cvModel = crossval.fit(tr)
    
    #output the parameters to results folder
    head = header.split(",")
    intercept = cvModel.bestModel.intercept
    coef = cvModel.bestModel.weights
    coef_file = open(resultDir_master + app_name + '_Coef.txt', "w")
    coef_file.writelines("Intercept, %f" %intercept)
    coef_file.writelines("\n")
    for id in range(len(coef)):
        coef_file.writelines("%s , %f" %(head[id+2] ,coef[id]))
        coef_file.writelines("\n")
    coef_file.close()

    # Predict on training data
    prediction_tr = cvModel.transform(tr)
    #prediction_tr2 = prediction_tr.withColumn('label', prediction_tr[
    # 'label'].cast('double'))#convert label to double
    pred_score_tr = prediction_tr.select('patid', 'label', 'probability','prediction')
    #predict on test data
    prediction_ts = cvModel.transform(ts)
    #prediction_ts2 = prediction_ts.withColumn('label', prediction_ts[
    # 'label'].cast('double'))#convert label to double
    pred_score_ts = prediction_ts.select('patid', 'label', 'probability','prediction')
    # AUC
    AUC_tr = evaluator.evaluate(prediction_tr,{evaluator.metricName:CVCriteria})
    AUC_ts = evaluator.evaluate(prediction_ts,{evaluator.metricName:CVCriteria})
    #print out AUC results
    auc = "Training data " + CVCriteria +" = %s " % round(AUC_tr, 4) + "\n" + \
          "Test data " + CVCriteria +" = %s " % round(AUC_ts, 4)   
    auc_file = open(resultDir_master + app_name + '_AUC.txt', "w")
    auc_file.writelines(auc)
    auc_file.close()
    
    #Identify the probility of response
    pred_score_tr1 = pred_score_tr.select(pred_score_tr.patid, 
                                         pred_score_tr.label,
                                         pred_score_tr.prediction,
                                         getitem(0)('probability').alias('p1'),
                                         getitem(1)('probability').alias('p2'))
    pred_score_ts1 = pred_score_ts.select(pred_score_ts.patid, 
                                         pred_score_ts.label,
                                         pred_score_ts.prediction,
                                         getitem(0)('probability').alias('p1'),
                                         getitem(1)('probability').alias('p2'))
                                         
    firstone = pred_score_tr1.take(1)
    firstonezip = zip(*firstone)
    p1 = firstonezip[3]
    p2 = firstonezip[4]
    pred = firstonezip[2][0]
    if ((p1 > p2 and pred == 1.0) or (p1 < p2 and pred == 0.0)):
        pred_score_tr1 = pred_score_tr1.withColumnRenamed('p1','Prob_1')
        pred_score_tr1 = pred_score_tr1.withColumnRenamed('p2','Prob_0')
        pred_score_ts1 = pred_score_ts1.withColumnRenamed('p1','Prob_1')
        pred_score_ts1 = pred_score_ts1.withColumnRenamed('p2','Prob_0')
    else:
        pred_score_tr1 = pred_score_tr1.withColumnRenamed('p1','Prob_0')
        pred_score_tr1 = pred_score_tr1.withColumnRenamed('p2','Prob_1')
        pred_score_ts1 = pred_score_ts1.withColumnRenamed('p1','Prob_0')
        pred_score_ts1 = pred_score_ts1.withColumnRenamed('p2','Prob_1')
 
    #output results to S3
    tr_lines = pred_score_tr1.map(toCSVLine)
    tr_lines.saveAsTextFile((resultDir_s3 + app_name + '_pred_tr'))
    ts_lines = pred_score_ts1.map(toCSVLine)
    ts_lines.saveAsTextFile((resultDir_s3 + app_name + '_pred_ts'))
    
if __name__ == "__main__":
 
    ErrCnt = parameter_ck()
    
    if (ErrCnt == 0):
        sc = SparkContext(appName = app_name)
        sqlContext = SQLContext(sc)
        main(sc)
        sc.stop()
    else:
        print('!!!Errors in parameters! alpha should be in range [0,1], lambda should be >= 0')
        
    



