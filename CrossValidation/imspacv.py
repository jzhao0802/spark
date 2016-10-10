from pyspark import SparkContext
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.sql.functions import lit
from pyspark import since, keyword_only
from pyspark.sql.types import IntegerType
import numpy

__all__ = ["CrossValidatorWithStratificationID"]


def _write_to_np_MetricValueEveryParamSet(metricValueEveryParamSet, 
                                          colnamesMetricValueEveryParamSet, 
                                          j, params, metric):
    #
    for paramNameStruct in params.keys():
        paramName = paramNameStruct.name
        paramVal = params[paramNameStruct]
        try:
            colID = colnamesMetricValueEveryParamSet.index(paramName)
        except ValueError:
            print("Error! " + paramName + " doesn't exist in the list of hyper-parameter names colnamesMetricValueEveryParamSet.")
        metricValueEveryParamSet[j, colID] = paramVal
        
    metricValueEveryParamSet[j, -1] += metric
    
    return metricValueEveryParamSet
    

class CrossValidatorWithStratificationID(CrossValidator):
    """
    Similar to the built-in pyspark.ml.tuning.CrossValidator, but the stratification will be done
    according to the fold IDs in a column of the input data. 
    """
    
    # a placeholder to make it appear in the generated doc
    stratifyCol = Param(Params._dummy(), "stratifyCol", "column name of the stratification ID")
    
    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None):
        """
        __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None)
        """     
        if stratifyCol is None:
            raise ValueError("stratifyCol must be specified.")        
        super(CrossValidatorWithStratificationID, self).__init__()
        self.bestIndex = None
        self.metricValueEveryParamSet = None
        self.colnamesMetricValueEveryParamSet = None
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)
        
    @keyword_only
    @since("1.4.0")
    def setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None):
        """
        setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None):
        Sets params for cross validator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)
        
    @keyword_only
    @since("1.4.0")
    def setNumFolds(self, value):
        """
        Deprecated.
        """
        raise AttributeError(type(self).__name__ + " does not have method 'setNumFolds'")
        
    @since("1.4.0")
    def getNumFolds(self):
        """
        Deprecated.
        """
        raise AttributeError(type(self).__name__ + " does not have method 'getNumFolds'")
    
    def _fit(self, dataset):
        try: 
            stratifyCol = self.getOrDefault(self.stratifyCol)
            dfStratifyCol = dataset.select(stratifyCol)
            rowsStratifyCol = dfStratifyCol.distinct().collect()
            foldIDs = [x[stratifyCol] for x in rowsStratifyCol]
            if (set(foldIDs) != set(range(max(foldIDs)+1))):
                raise ValueError("The stratifyCol column does not have zero-based consecutive integers as fold IDs.")
        except Exception as e:
            print("Something is wrong with the stratifyCol:")
            raise
        
        # 
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        
        nFolds = dfStratifyCol.distinct().count()
        
        # select features, label and foldID in order
        featuresCol = est.getFeaturesCol()
        labelCol = est.getLabelCol()
        dataWithFoldID = dataset.select(featuresCol, labelCol, stratifyCol)
        
        paramNames = [x.name for x in epm[0].keys()]
        metricValueCol = "metricValue"
        self.metricValueEveryParamSet = numpy.empty((len(epm), 2 + len(paramNames)))
        self.metricValueEveryParamSet[:,0] = numpy.arange(len(epm))
        self.metricValueEveryParamSet[:,-1] = 0
        self.colnamesMetricValueEveryParamSet = ["paramSetID"] + paramNames + [metricValueCol]
        
        for i in range(nFolds):
            condition = (dataWithFoldID[stratifyCol] == i)    
            validation = dataWithFoldID.filter(condition).select(featuresCol, labelCol)
            train = dataWithFoldID.filter(~condition).select(featuresCol, labelCol)            
            
            for j in range(numModels):
                model = est.fit(train, epm[j])
                # TODO: duplicate evaluator to take extra params from input
                metric = eva.evaluate(model.transform(validation, epm[j]))
                
                self.metricValueEveryParamSet = \
                    _write_to_np_MetricValueEveryParamSet(self.metricValueEveryParamSet, 
                                                          self.colnamesMetricValueEveryParamSet, 
                                                          j, epm[j], metric)               
        
        self.metricValueEveryParamSet[:,-1] = self.metricValueEveryParamSet[:,-1] / nFolds
        
        if eva.isLargerBetter():
            self.bestIndex = numpy.argmax(self.metricValueEveryParamSet[:,-1])
        else:
            self.bestIndex = numpy.argmin(self.metricValueEveryParamSet[:,-1])
        
        #return the best model
        self.bestModel = est.fit(dataset, epm[self.bestIndex])
        return CrossValidatorModel(self.bestModel)
        
    #returns the hyperparameters of the best model chosen by the cross validator
    def getBestModelParams(self):
        epm = self.getOrDefault(self.estimatorParamMaps)
        
        if (self.bestIndex is not None):
            bestModelParms = dict((key.name, value) for key, value in epm[self.bestIndex].iteritems())
        else:
            bestModelParms = "\nCrossvalidation has not run yet.\n"
        return bestModelParms
        
    def getCVMetrics(self):
        if self.metricValueEveryParamSet is None: 
            raise ValueError("metrics has the value None. Make sure the object of CrossValidatorNoStratification is fitted before calling the getCVMetrics method. ")
            
        if len(self.colnamesMetricValueEveryParamSet) != self.metricValueEveryParamSet.shape[1]:
            raise ValueError("Number of column names self.colnamesMetricValueEveryParamSet don't match the number of columns self.metricValueEveryParamSet. ")
            
        # convert to pyspark.sql.DataFrame
        metricsAsList = [tuple(float(y) for y in x) for x in self.metricValueEveryParamSet]
        df = SQLContext.getOrCreate(SparkContext.getOrCreate()).createDataFrame(metricsAsList, self.colnamesMetricValueEveryParamSet)
        df = df.withColumn(df.columns[0], df[df.columns[0]].cast(IntegerType()))
        
        return df

        
import unittest
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row

class CrossValidatorWithStratificationIDTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession\
                   .builder\
                   .appName("CrossValidatorWithStratificationIDTests")\
                   .getOrCreate()
        dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/datasetWithFoldID.csv"
        cls.data = cls.spark\
                  .read\
                  .option("header", "true")\
                  .option("inferSchema", "true")\
                  .csv(dataFileName)        
        cls.data.cache()
        
        
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        
    def test_CVResult(self):
        assembler = VectorAssembler(inputCols=self.data.columns[1:(-1)], outputCol="features")
        stratifyCol = "foldID"
        featureAssembledData = assembler.transform(self.data).select("y", "features", stratifyCol)   
        lr = LinearRegression(maxIter=1e5, standardization=True, 
                              featuresCol="features", labelCol="y")
        evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y")
        lambdas = [0.1,1]
        alphas = [0,0.5]
        paramGrid = ParamGridBuilder()\
               .addGrid(lr.regParam, lambdas)\
               .addGrid(lr.elasticNetParam, alphas)\
               .build()   
        validator = CrossValidatorWithStratificationID(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              stratifyCol=stratifyCol)
        cvModel = validator.fit(featureAssembledData)
        metrics = validator.getCVMetrics()
        collectedMetrics = metrics.collect()
        def localRoundMetricValue(rr):
            rrAsDict = rr.asDict()
            rrAsDict["metricValue"] = round(rrAsDict["metricValue"], 3)
            return rrAsDict
        roundedMetrics = [localRoundMetricValue(x) for x in collectedMetrics]
        def localConvertDictToStr(dd):
            return "".join("{}:{};".format(key,value) for key, value in dd.items())
        strMetrics = set([localConvertDictToStr(x) for x in roundedMetrics])
        
        expectedMetrics = set([\
            "regParam:0.1;elasticNetParam:0.0;paramSetID:0;metricValue:1.087;",
            "regParam:1.0;elasticNetParam:0.0;paramSetID:1;metricValue:1.052;",
            "regParam:0.1;elasticNetParam:0.5;paramSetID:2;metricValue:1.06;",
            "regParam:1.0;elasticNetParam:0.5;paramSetID:3;metricValue:1.069;"
        ])
        
        self.assertEqual(strMetrics, expectedMetrics, "Incorrect list of evaluation metric values for all hyper-parameter sets.")
        
        bestParams = validator.getBestModelParams()
        self.assertEqual(bestParams, {'regParam': 1, 'elasticNetParam': 0}, "Incorrect best parameters.")
        
            
if __name__ == "__main__":
    
    unittest.main()
    