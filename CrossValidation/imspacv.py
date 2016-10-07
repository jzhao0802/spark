from pyspark import SparkContext
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.sql.functions import lit
from pyspark import keyword_only
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
        
    metricValueEveryParamSet[j, -1] = metric
    
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
            dfStratifyCol = dataset.select(self.stratifyCol)
            rowsStratifyCol = dfStratifyCol.distinct().collect()
            foldIDs = [x[self.stratifyCol] for x in rowsStratifyCol]
            if (set(foldIDs) != set(range(max(foldIDs))))
                raise ValueError("The stratifyCol column does not have zero-based consecutive integers as fold IDs.")
        except Exception as e:
            print("Something is wrong with the stratifyCol": ")
            print(e)
        
        # 
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        
        nFolds = dfStratifyCol.distinct().count()
        
    
        # select features, label and foldID in order
        featuresCol = est.getFeaturesCol()
        labelCol = est.getLabelCol()
        dataWithFoldID = dataset.select(featuresCol, labelCol, self.stratifyCol)
        
        paramNames = [x.name for x in emp[0].keys()]
        metricValueCol = "metricValue"
        self.metricValueEveryParamSet = numpy.empty((len(epm), 2 + len(paramNames)))
        self.colnamesMetricValueEveryParamSet = ["paramSetID"] + paramNames + [metricValueCol]
    
        for i in range(nFolds):
            condition = (dataWithFoldID[self.stratifyCol] == i)    
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
            self.bestIndex = np.argmax(self.metricValueEveryParamSet[:,-1])
        else:
            self.bestIndex = np.argmin(self.metricValueEveryParamSet[:,-1])
        
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
        df = SQLContext.getOrCreate().createDataFrame(metricsAsList, self.colnamesMetricValueEveryParamSet)
        
        return df
