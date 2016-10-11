import os
import time
import datetime
import random
import numpy as np
from pyspark import SparkContext
from pyspark.ml.param import Params, Param
from pyspark.ml import Estimator
from pyspark import keyword_only
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
#from pyspark.mllib.common import _py2java, _java2py
from stratification import AppendDataMatchingFoldIDs

__all__ = ['CrossValidatorWithStratification']

class CrossValidatorWithStratification(Estimator):
    """
    K-fold cross validation considering data matching (every positive
    matches the same number of negative data; every negative matches
    with one and only one positive)

    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.evaluation import BinaryClassificationEvaluator
    >>> from pyspark.mllib.linalg import Vectors
    >>> dataset = sqlContext.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0),
    ...      (Vectors.dense([0.4]), 1.0),
    ...      (Vectors.dense([0.5]), 0.0),
    ...      (Vectors.dense([0.6]), 1.0),
    ...      (Vectors.dense([1.0]), 1.0)] * 10,
    ...     ["features", "label"])
    >>> lr = LogisticRegression()
    >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    >>> evaluator = BinaryClassificationEvaluator()
    >>> cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
    >>> cvModel = cv.fit(dataset)
    >>> evaluator.evaluate(cvModel.transform(dataset))
    0.8333...
    """
    
    # a placeholder to make it appear in the generated doc
    estimator = Param(Params._dummy(), "estimator", "estimator to be cross-validated")
    
    # a placeholder to make it appear in the generated doc
    estimatorParamMaps = Param(Params._dummy(), "estimatorParamMaps", "estimator param maps")
    
    # a placeholder to make it appear in the generated doc
    evaluator = Param(
        Params._dummy(), "evaluator",
        "evaluator used to select hyper-parameters that maximize the cross-validated metric")
    
    # a placeholder to make it appear in the generated doc
    numFolds = Param(Params._dummy(), "numFolds", "number of folds for cross validation")
    
    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3):
        """
        __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3)
        """
        super(CrossValidatorWithStratification, self).__init__()
        #: param for estimator to be cross-validated
        self.estimator = Param(self, "estimator", "estimator to be cross-validated")
        #: param for estimator param maps
        self.estimatorParamMaps = Param(self, "estimatorParamMaps", "estimator param maps")
        #: param for the evaluator used to select hyper-parameters that
        #: maximize the cross-validated metric
        self.evaluator = Param(
            self, "evaluator",
            "evaluator used to select hyper-parameters that maximize the cross-validated metric")
        #: param for number of folds for cross validation
        self.numFolds = Param(self, "numFolds", "number of folds for cross validation")
        self._setDefault(numFolds=3)
        self.bestIndex = None
        self.metrics = None
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)
    
    @keyword_only
    def setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3):
        """
        setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3):
        Sets params for cross validator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)
    
    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`.
        """
        self._paramMap[self.estimator] = value
        return self
    
    def getEstimator(self):
        """
        Gets the value of estimator or its default value.
        """
        return self.getOrDefault(self.estimator)
    
    def setEstimatorParamMaps(self, value):
        """
        Sets the value of :py:attr:`estimatorParamMaps`.
        """
        self._paramMap[self.estimatorParamMaps] = value
        return self
    
    def getEstimatorParamMaps(self):
        """
        Gets the value of estimatorParamMaps or its default value.
        """
        return self.getOrDefault(self.estimatorParamMaps)
    
    def setEvaluator(self, value):
        """
        Sets the value of :py:attr:`evaluator`.
        """
        self._paramMap[self.evaluator] = value
        return self
    
    def getEvaluator(self):
        """
        Gets the value of evaluator or its default value.
        """
        return self.getOrDefault(self.evaluator)
    
    def setNumFolds(self, value):
        """
        Sets the value of :py:attr:`numFolds`.
        """
        self._paramMap[self.numFolds] = value
        return self
    
    def getNumFolds(self):
        """
        Gets the value of numFolds or its default value.
        """
        return self.getOrDefault(self.numFolds)
    
    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        idColName = "foldID"
        dataWithFoldID = AppendDataMatchingFoldIDs(data=dataset, nFolds=nFolds)
        dataWithFoldID.cache()
        self.metrics = np.zeros(numModels)
    
        # select features, label and foldID in order
        dataWithFoldID = dataWithFoldID.select('features', 'label', 'foldID')
    
        for i in range(nFolds):
            condition = (dataWithFoldID[idColName] == i)    
            validation = dataWithFoldID.filter(condition).select('features', 'label')
            train = dataWithFoldID.filter(~condition).select('features', 'label')
            stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            si_model = stringIndexer.fit(train)
            train_td = si_model.transform(train)
            validation_td = si_model.transform(validation)
            
            for j in range(numModels):
                model = est.fit(train_td, epm[j])
                # TODO: duplicate evaluator to take extra params from input
                metric = eva.evaluate(model.transform(validation_td, epm[j]))
                self.metrics[j] += metric
        
        if eva.isLargerBetter():
            self.bestIndex = np.argmax(self.metrics)
        else:
            self.bestIndex = np.argmin(self.metrics)
        
        #return the best model
        self.bestModel = est.fit(dataset, epm[self.bestIndex])
        return CrossValidatorModel(self.bestModel)
    
    #returns the hyperparameters of the best model chosen by the cross validator
    def getBestModelParms(self):
        epm = self.getOrDefault(self.estimatorParamMaps)
        
        if (self.bestIndex is not None):
            bestModelParms = dict((key.name, value) for key, value in epm[self.bestIndex].iteritems())
        else:
            bestModelParms = "\nCrossvalidation has not run yet.\n"
        return bestModelParms
        
    def getCVMetrics(self):
        if self.metrics is None: 
            raise TypeError("metrics has the value None. Make sure the object of CrossValidatorWithStratification is fitted before calling the getCVMetrics method. ")
        
        return self.metrics
    
    def copy(self, extra=None):
        if extra is None:
            extra = dict()
        newCV = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newCV.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMaps remain the same
        if self.isSet(self.evaluator):
            newCV.setEvaluator(self.getEvaluator().copy(extra))
        return newCV

def _Test():
    from pyspark import SQLContext

    #input parameters
    pos_file = "dat_hae.csv"
    neg_file = "dat_nonhae.csv"
    data_file = "dat_results.csv"
    start_tree = 5
    stop_tree = 10
    num_tree = 2
    start_depth = 2
    stop_depth = 3
    num_depth = 2
    nFolds = 3

    #creating SparkContext and HiveContext
    sc = SparkContext(appName="Test")
    sqlContext = SQLContext(sc)

    s3_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/"
    data_path = "s3://emr-rwes-pa-spark-dev-datastore/Hui/datafactz_test/01_data/"
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = s3_path + "Results/" + st + "/"
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3)
    #resultDir_master = "~/Downloads/datafactz/task_3/toydata/Results/" + st + "/"
    #if not os.path.exists(resultDir_master):
    #    os.makedirs(resultDir_master)

    # seed
    seed = 42
    random.seed(seed)

    #reading in the data from S3
    pos = sqlContext.read.load((data_path + pos_file),
                              format='com.databricks.spark.csv',
                              header='true',
                              inferSchema='true')

    neg = sqlContext.read.load((data_path + neg_file),
                              format='com.databricks.spark.csv',
                              header='true',
                              inferSchema='true')
    #get the column names
    pos_col = pos.columns
    neg_col = neg.columns

    #combine features
    assembler_pos = VectorAssembler(inputCols=pos_col[2:], outputCol="features")
    assembler_neg = VectorAssembler(inputCols=neg_col[2:-1], outputCol="features")

    #get the input positive and negative dataframe
    pos_asmbl = assembler_pos.transform(pos)\
                .select('PATIENT_ID', 'HAE', 'features')\
                .withColumnRenamed('PATIENT_ID', 'matched_positive_id')\
                .withColumnRenamed('HAE', 'label')

    pos_ori = pos_asmbl.withColumn('label', pos_asmbl['label'].cast('double'))\
                .select('matched_positive_id', 'label', 'features')

    neg_asmbl = assembler_neg.transform(neg)\
                .select('HAE', 'HAE_PATIENT_ID', 'features')\
                .withColumnRenamed('HAE', 'label')\
                .withColumnRenamed('HAE_PATIENT_ID', 'matched_positive_id')

    neg_ori = neg_asmbl.withColumn('label', neg_asmbl['label'].cast('double')) \
                .select('matched_positive_id', 'label', 'features')

    data = pos_ori.unionAll(neg_ori)

    dataWithFoldID = AppendDataMatchingFoldIDs(data=data, nFolds=nFolds)
    dataWithFoldID.cache()

    # iteration through all folds
    for iFold in range(nFolds):
        # stratified sampling
        ts = dataWithFoldID.filter(dataWithFoldID.foldID == iFold)
        tr = dataWithFoldID.filter(dataWithFoldID.foldID != iFold)

        # remove the fold id column
        keep = [c for c in ts.columns if c != "foldID"]
        ts = ts.select(*keep)
        tr = tr.select(*keep)

        # transfer to RF invalid label column
        stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
        si_model = stringIndexer.fit(tr)
        tr_td = si_model.transform(tr)
        ts_td = si_model.transform(ts)

        # Build the model
        rf = RandomForestClassifier(labelCol="indexed", featuresCol="features")

        # Create the parameter grid builder
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, list(np.linspace(start_tree, stop_tree,
                                                   num_tree).astype('int'))) \
            .addGrid(rf.maxDepth, list(np.linspace(start_depth, stop_depth,
                                                   num_depth).astype('int'))) \
            .build()

        # Create the evaluator
        evaluator = BinaryClassificationEvaluator(labelCol="indexed")

        # Create the cross validator
        crossval = CrossValidatorWithStratification( estimator=rf,
                                                     estimatorParamMaps=paramGrid,
                                                     evaluator=evaluator,
                                                     numFolds=nFolds)

        # run cross-validation and choose the best parameters
        cvModel = crossval.fit(tr_td)

        # Predict on training data
        prediction_tr = cvModel.transform(tr_td)
        pred_score_tr = prediction_tr.select('label', 'indexed', 'probability')

        # predict on test data
        prediction_ts = cvModel.transform(ts_td)
        pred_score_ts = prediction_ts.select('label', 'indexed', 'probability')

        # AUC
        AUC_tr = evaluator.evaluate(prediction_tr, {evaluator.metricName: 'areaUnderROC'})
        AUC_ts = evaluator.evaluate(prediction_ts, {evaluator.metricName: 'areaUnderROC'})

        # print out results
        #fAUC = open(resultDir_master + "AUC_fold" + str(iFold) + ".txt", "a")
        #fAUC.write("{}: Traing AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_tr))
        #fAUC.write("{}: Test AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_ts))
        #fAUC.close()

        pred_score_tr.coalesce(1)\
                            .write.format('com.databricks.spark.csv')\
                            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_tr_fold" + str(iFold) + ".csv")
        pred_score_ts.coalesce(1)\
                            .write.format('com.databricks.spark.csv')\
                            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_ts_fold" + str(iFold) + ".csv")

    #fFinished = open(resultDir_master + "finished.txt", "a")
    #fFinished.write("Test for {} finished. Please manually check the result.. \n".format(data_file))
    #fFinished.close()

    print crossval.getBestModelParms()

    print("------------------------------------------")
    print("--- PROGRAM COMPLETED SUCCESSFULLY !!! ---")
    print("------------------------------------------")


if __name__ == "__main__":
    # some test
    _Test()