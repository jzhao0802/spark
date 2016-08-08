import numpy as np
from pyspark import SparkContext, SQLContext
from pyspark.sql.functions import explode, count

import unittest

def AppendDataMatchingFoldIDs(data, nFolds, nDesiredPartitions="None"):
    """
    Select three columns ("matched_positive_id", "label", "features") 
    and then append a column of stratified fold IDs to the input 
    DataFrame.

    Parameters
    ----------
    data: pyspark.sql.dataframe.DataFrame
        The input DataFrame.
    labelCol: string
        The column name of the label. Default: "label".
    nFolds: integer
        The number of foldds to stratify.
    nDesiredPartitions: integer or "None".
        The number of partitions in the returned DataFrame. If "None",
        the result has the same number of partitions as the input data.
        Default: "None".

    Returns
    ----------
    A pyspark.sql.DataFrame with 4 columns: three columns 
    ("matched_positive_id", "label", "features") from the original 
    DataFrame and a column appended to the input data
    as the fold ID. The column name of the fold ID is "foldID".
    """
    sc = SparkContext._active_spark_context
    sqlContext = SQLContext(sc)
    
    # selecting the required columns (dropping the 'indexed' column)
    data = data.select('matched_positive_id', 'label', 'features')
    
    if nDesiredPartitions == "None":
        nDesiredPartitions = data.rdd.getNumPartitions()
    
    # Group by key, where key is matched_positive_id
    data_rdd = data.rdd.map(lambda (x, y, z): (x, (y, z))) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1])))
    
    # getting the count of positive after grouping
    nPoses = data_rdd.count()
    npFoldIDsPos = np.array(list(range(nFolds)) * np.ceil(float(nPoses) / nFolds))
    
    # select the actual numbers of FoldIds matching the count of positive data points
    npFoldIDs = npFoldIDsPos[:nPoses]
    
    # Shuffle the foldIDs to give randomness
    np.random.shuffle(npFoldIDs)
    
    rddFoldIDs = sc.parallelize(npFoldIDs, nDesiredPartitions).map(int)
    dfDataWithIndex = data_rdd.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "orgData")
    dfNewKeyWithIndex = rddFoldIDs.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "key")
    dfJoined = dfDataWithIndex.join(dfNewKeyWithIndex, "_2") \
        .select('orgData._1', 'orgData._2', 'key') \
        .withColumnRenamed('key', 'foldID') \
        .coalesce(nDesiredPartitions)
    
    """explding the features and label column,
     which means grouped data of labels and features will be expanded.
     In short, grouped data by matched_positive_id will be expanded."""
    dfExpanded = dfJoined.select(dfJoined._1, explode(dfJoined._2).alias("label_features"), dfJoined.foldID)
    
    # selecting the column with required meaningful names
    dfWithFoldID = dfExpanded.select(dfExpanded["_1"].alias("matched_positive_id"), 
                                     dfExpanded["label_features"]["_2"].alias("features"), 
                                     dfExpanded["foldID"], 
                                     dfExpanded["label_features"]["_1"].alias("label"))
    
    return dfWithFoldID
    
def areFoldIdsEqual(ids):
    if len(set(ids)) == 1:
        return True
    else:
        return False

class StratificationTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sc = SparkContext(appName=cls.__name__)
        cls.sqlContext = SQLContext(cls.sc)
        cls.testData = cls.sqlContext.read.load("s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/matchedID_label_features.csv",
                              format='com.databricks.spark.csv',
                              header='true',
                              inferSchema='true')
        cls.testData.cache()
        # cls.nData = cls.testData.count()
        
    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()
        
    def GetNumDistinctMatchedPosIDs(self, df):
        return df.select("foldID", "matched_positive_id")\
                        .rdd\
                        .groupByKey()\
                        .map(lambda x: len(set(x[1])))\
                        .collect()
    
    def test_Stratification(self):
        # 5 different matched_positive_ids
        # 2 folds
        nFolds = 2
        dataWithFoldID_2 = AppendDataMatchingFoldIDs(data=StratificationTests.testData, nFolds=nFolds)
        # those with the same matched_positive_id should have the same fold ID
        
        self.assertTrue(\
            dataWithFoldID_2.select("matched_positive_id", "foldID")\
                        .rdd\
                        .groupByKey()\
                        .map(lambda x: areFoldIdsEqual(list(x[1])))\
                        .reduce(lambda x,y: x&y),
            "Data with the same matched_positive_id should have the same fold ID!"
        )
        
        # one fold should have two different matched_positive_ids, the other 3
        nMatchedPosIDs = self.GetNumDistinctMatchedPosIDs(dataWithFoldID_2)
        self.assertTrue(\
            (nMatchedPosIDs == [2,3]) | (nMatchedPosIDs == [3,2]),
            "When split into 2 folds, one fold must have two distinct matched_positive_ids and the other must have 3."
        )
        
        # the union of the distinct matched_positive_ids should cover all those in the data
        setMatchedPositiveIDs = dataWithFoldID_2.select("foldID", "matched_positive_id")\
                        .rdd\
                        .groupByKey()\
                        .map(lambda x: set(x[1]))\
                        .reduce(lambda x,y: x.union(y))
        self.assertEqual(
            setMatchedPositiveIDs,
            set(StratificationTests.testData.select("matched_positive_id").rdd.map(lambda x: x.matched_positive_id).collect()),
            "The union of the distinct matched_positive_ids in all folds should cover all those in the data. "
        )
        
        # one matched_positive_id for each fold
        nFolds = 5
        dataWithFoldID_5 = AppendDataMatchingFoldIDs(data=StratificationTests.testData, nFolds=nFolds)
        nMatchedPosIDs = self.GetNumDistinctMatchedPosIDs(dataWithFoldID_5)
        self.assertTrue(\
            nMatchedPosIDs == [1]*5,
            "When split into 5 folds, one fold must have only one distinct matched_positive_ids."
        )
        
        
        # two matched_positive_ids for each fold
        nFolds = 2
        dataWithFoldID_22 = \
            AppendDataMatchingFoldIDs(\
                data=StratificationTests.testData.filter(StratificationTests.testData.matched_positive_id != 0), 
                nFolds=nFolds
            )
        nMatchedPosIDs = self.GetNumDistinctMatchedPosIDs(dataWithFoldID_22)
        self.assertTrue(\
            nMatchedPosIDs == [2]*2,
            "Now the data has only 4 distinct values for matched_positive_ids. When split into 2 folds, each fold should have 2 distinct values."
        )
            
if __name__ == "__main__":
    
    unittest.main()
    