from imspaeva import BinaryClassificationEvaluatorWithPrecisionAtRecall
from pyspark.sql import SparkSession
import unittest
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit, col
from pyspark.ml.linalg import Vectors


def assemble_pred_vector(data, orgLabelCol, orgPositivePredictionCol, newLabelCol, newPredictionCol):
    newdata = data \
        .withColumn('prob_0', lit(1) - data[orgPositivePredictionCol]) \
        .withColumnRenamed(orgPositivePredictionCol, 'prob_1') \
        .withColumnRenamed(orgLabelCol, newLabelCol)
    asmbl = VectorAssembler(inputCols=['prob_0', 'prob_1'],
                            outputCol=newPredictionCol)
    # get the input positive and negative dataframe
    data_asmbl = asmbl.transform(newdata).select(newLabelCol, newPredictionCol)

    return data_asmbl

class PREvaluationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName(cls.__name__).getOrCreate()
        file = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/task6/labelPred.csv"
        scoreAndLabels = cls.spark\
                  .read\
                  .option("header", "true")\
                  .option("inferSchema", "true")\
                  .csv(file)
        scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double'), scoreAndLabels.pred)\
                                       .withColumnRenamed("cast(label as double)", "label")
        cls.rawPredictionCol = "pred"
        cls.labelCol = "label"
        cls.scoreAndLabelsVectorised = assemble_pred_vector(data=scoreAndLabels,
                                                            orgLabelCol="label",
                                                            orgPositivePredictionCol="pred",
                                                            newLabelCol=cls.labelCol,
                                                            newPredictionCol=cls.rawPredictionCol)
        cls.scoreAndLabelsVectorised.cache()
        cls.tolerance = 0.0050

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        

    def test_areaUnderROC(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised, {evaluator.metricName: 'areaUnderROC'})
        self.assertTrue((0.8290 - self.tolerance) <= ROC <= (0.8290 + self.tolerance), "Area under ROC value is incorrect.")

    def test_ROC_isLargeBetter(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall()
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning False.")

    def test_areaUnderPR(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        PR = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised, {evaluator.metricName: 'areaUnderPR'})
        self.assertTrue((0.8372 - self.tolerance) <= PR <= (0.8372 + self.tolerance), "Area under PR value is incorrect.")

    def test_PR_isLargeBetter(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall()
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")

    def test_is_precision_matching_1(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        desiredRecall = 0.2
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionAtGivenRecall",
                                        "metricParams": {"recallValue": desiredRecall}})
        self.assertEqual(round(precision, 4), 1.0, "precisionAtGivenRecall metric producing incorrect precision: {}".format(precision))

    def test_is_precision_matching_2(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        desiredRecall = 0.4
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionAtGivenRecall",
                                        "metricParams": {"recallValue": desiredRecall}})
        self.assertEqual(round(precision, 4), 1, "precisionAtGivenRecall metric producing incorrect precision: {}".format(precision))

    def test_is_precision_matching_3(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        desiredRecall = 0.6
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionAtGivenRecall",
                                        "metricParams": {"recallValue": desiredRecall}})
        self.assertEqual(round(precision, 4), 0.8000, "precisionAtGivenRecall metric producing incorrect precision: {}".format(precision))

    def test_is_precision_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false before calculating precision at recall.")
        desiredRecall = 0.2
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised,
                                       {"metricName": "precisionAtGivenRecall",
                                        "metricParams": {"recallValue": desiredRecall}})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false after calculating precision at recall.")

    def test_precision_at_given_recall_with_init(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(metricName="precisionAtGivenRecall", rawPredictionCol=self.rawPredictionCol,
                                                        labelCol=self.labelCol, metricParams={"recallValue": 0.6})
        precision = evaluator.evaluate(self.scoreAndLabelsVectorised)
        self.assertEqual(round(precision, 4), 0.8, "Incorrect precision result at the given recall using init")

    def test_precision_at_given_recall_with_evaluate(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        precision = evaluator.evaluate(
            PREvaluationMetricTests.scoreAndLabelsVectorised,
            {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.6}})

        self.assertEqual(round(precision, 4), 0.8, "Incorrect precision result at the given recall using evaluate")

    def test_is_ROC_matching(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol)
        ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised, {evaluator.metricName: 'areaUnderROC'})
        self.assertTrue((0.8290 - self.tolerance) <= ROC<= (0.8290 + self.tolerance), "ROC value is outside of the specified range")

    def test_precision_at_given_recall_different_colnames(self):
        new_label_col = "AA"
        new_prediction_col = "BB"
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(metricName="precisionAtGivenRecall", rawPredictionCol=new_prediction_col,
                                                        labelCol=new_label_col, metricParams={"recallValue": 0.6})
        precision = evaluator.evaluate(
            self.scoreAndLabelsVectorised\
                .withColumnRenamed(self.labelCol, new_label_col)\
                .withColumnRenamed(self.rawPredictionCol, new_prediction_col))
        self.assertEqual(round(precision, 4), 0.8, "Incorrect precision result at the given recall using init")

    def test_setParams_ROC(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="xx", labelCol="yy", metricName="areaUnderPR")
        evaluator.setParams(rawPredictionCol=self.rawPredictionCol, labelCol=self.labelCol, metricName="areaUnderROC")
        ROC = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised)
        self.assertTrue((0.8290 - self.tolerance) <= ROC <= (0.8290 + self.tolerance),
                        "ROC value is outside of the specified range")

    def test_setParams_precisionGivenRecall(self):
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="xx", labelCol="yy", metricName="areaUnderPR")
        evaluator.setParams(
            rawPredictionCol=self.rawPredictionCol,
            labelCol=self.labelCol,
            metricName="precisionAtGivenRecall",
            metricParams={"recallValue":0.4}
        )
        precision = evaluator.evaluate(PREvaluationMetricTests.scoreAndLabelsVectorised)
        self.assertEqual(round(precision, 4), 1,
                         "precisionAtGivenRecall metric producing incorrect precision: {}".format(precision))

    def test_simple(self):
        scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
                             [(0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0)])
        dataset = PREvaluationMetricTests.spark.createDataFrame(scoreAndLabels, ["raw", "label"])
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(
            rawPredictionCol="raw",
            metricName="precisionAtGivenRecall",
            metricParams={"recallValue":1}
        )
        precision = evaluator.evaluate(dataset)
        self.assertEqual(round(precision, 4), 0.5714,
                         "precisionAtGivenRecall metric producing incorrect precision: {}".format(precision))
                         
    def test_label_order(self):
        scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
                             [(0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0),
                             (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0),
                             (0.1, 0.0), (0.1, 0.0), (0.1, 0.0), (0.1, 0.0),
                             (0.1, 0.0), (0.1, 0.0), (0.1, 0.0), (0.1, 0.0),
                             (0.4, 1.0), (0.4, 1.0), (0.4, 1.0), (0.4, 1.0),
                             (0.4, 1.0), (0.4, 1.0), (0.4, 1.0), (0.4, 1.0),
                             (0.4, 0.0), (0.4, 0.0), (0.4, 0.0), (0.4, 0.0),
                             (0.4, 0.0), (0.4, 0.0), (0.4, 0.0), (0.4, 0.0),
                             (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                             (0.8, 1.0), (0.8, 1.0), (0.8, 1.0), (0.8, 1.0),
                             (0.8, 0.0), (0.8, 0.0), (0.8, 0.0), (0.8, 0.0),
                             (0.8, 0.0), (0.8, 0.0), (0.8, 0.0), (0.8, 0.0),])
        dataset = PREvaluationMetricTests.spark.createDataFrame(scoreAndLabels, ["raw", "label"])
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(
            rawPredictionCol="raw",
            metricName="precisionAtGivenRecall",
            metricParams={"recallValue":1}
        )
        precision = evaluator.evaluate(dataset)
        self.assertEqual(round(precision, 4), 0.5,
                         "precisionAtGivenRecall metric producing incorrect precision: {}".format(precision))

    def test_all_metrics_default(self):
        scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
                             [(0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0)])
        dataset = PREvaluationMetricTests.spark.createDataFrame(scoreAndLabels, ["raw", "label"])
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="raw")
        all_metrics = evaluator.evaluateWithSeveralMetrics(dataset)
        rounded_metrics = [{list(x.items())[0][0]: round(list(x.items())[0][1], 4)} for x in all_metrics]
        expected_metrics = [{"areaUnderROC": 0.7083}, 
                            {"areaUnderPR": 0.7083}, 
                            {"precisionAtGivenRecall at recallValue 0.05": 1.0000}]
        self.assertEqual(rounded_metrics, expected_metrics,
                         "Expected metrics: {0}; Result rounded metrics: {1};".format(expected_metrics, rounded_metrics))
                         
    def test_all_metrics_two_precisions(self):
        scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
                             [(0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0)])
        dataset = PREvaluationMetricTests.spark.createDataFrame(scoreAndLabels, ["raw", "label"])
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="raw")
        metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.05}},
                      {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 1}}]
        all_metrics = evaluator.evaluateWithSeveralMetrics(dataset, metricSets = metricSets)
        rounded_metrics = [{list(x.items())[0][0]: round(list(x.items())[0][1], 4)} for x in all_metrics]
        expected_metrics = [{"precisionAtGivenRecall at recallValue 0.05": 1.0000},
                            {"precisionAtGivenRecall at recallValue 1": 0.5714}]
        self.assertEqual(rounded_metrics, expected_metrics,
                         "Expected metrics: {0}; Result rounded metrics: {1};".format(expected_metrics, rounded_metrics))

    def test_pr_curve(self):
        preds = [0.249,  0.795,  0.178,  0.342,  0.6  ,  0.232,  0.511,  0.437,
                             0.249,  0.418,  0.56 ,  0.269,  0.224,  0.343,  0.201,  0.136,
                             0.249,  0.344,  0.56,  0.356,  0.286,  0.277,  0.296,  0.442,
                             0.249,  0.259,  0.397,  0.47 ,  0.726,  0.709,  0.447,  0.378,
                             0.455,  0.843,  0.745,  0.785,  0.318,  0.234,  0.47 ,  0.247]
        labels = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                              1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0]
        
        pp = [1.0, 0.0, 0.5, 0.667, 0.75, 0.8, 0.833, 0.857, 0.889, 0.9, 
        0.917, 0.846, 0.857, 0.867, 0.875, 0.824, 0.778, 0.737, 0.7, 0.667, 
        0.636, 0.609, 0.583, 0.56, 0.538, 0.556, 0.536, 0.517, 0.455, 0.441, 
        0.429, 0.444, 0.432, 0.421, 0.41, 0.4]
        rr = [0.0, 0.0, 0.063, 0.125, 0.188, 0.25, 0.313, 0.375, 0.5, 0.563, 
        0.688, 0.688, 0.75, 0.813, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 
        0.875, 0.875, 0.875, 0.875, 0.875, 0.938, 0.938, 0.938, 0.938, 0.938, 
        0.938, 1.0, 1.0, 1.0, 1.0, 1.0]
        n = len(pp)
        expected_curve = [{"precision":pp[ii],"recall":rr[ii]} for ii in range(n)]
                
        l1 = list(zip(preds,labels))
        l2 = [(Vectors.dense([1.0 - x[0], x[0]]), x[1]) for x in l1]
        dataset = PREvaluationMetricTests.spark.createDataFrame(l2, ["raw","label"])
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="raw")
        curve = evaluator._cal_pr_curve(dataset).coalesce(1).collect()
        curve_dict = [x.asDict() for x in curve]
        
        self.assertEqual(n, len(curve_dict), "Expected number of PR pairs: {1}; Result number of PR pairs: {0};".format(n, len(curve_dict)))            
        
        for i_pair in range(len(curve_dict)):
            for key,value in curve_dict[i_pair].items():
                curve_dict[i_pair][key] = round(value,3)
            self.assertTrue(curve_dict[i_pair] in expected_curve, "Result pair {0} is not in the expected PR pairs {1}.".format(curve_dict[i_pair],expected_curve))
        
        for i_pair in range(len(curve_dict)):
            self.assertTrue(expected_curve[i_pair] in curve_dict, "Expected PR pair {0} is not in the result PR pairs {1}.".format(expected_curve[i_pair],curve_dict))
            
    def test_precision_max_in_pr_pairs_with_same_recall(self):
        preds = [0.249,  0.795,  0.178,  0.342,  0.6  ,  0.232,  0.511,  0.437,
                             0.249,  0.418,  0.56 ,  0.269,  0.224,  0.343,  0.201,  0.136,
                             0.249,  0.344,  0.56,  0.356,  0.286,  0.277,  0.296,  0.442,
                             0.249,  0.259,  0.397,  0.47 ,  0.726,  0.709,  0.447,  0.378,
                             0.455,  0.843,  0.745,  0.785,  0.318,  0.234,  0.47 ,  0.247]
        labels = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                              1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0]
        l1 = list(zip(preds,labels))
        l2 = [(Vectors.dense([1.0 - x[0], x[0]]), x[1]) for x in l1]
        dataset = PREvaluationMetricTests.spark.createDataFrame(l2, ["raw","label"])
        
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="raw")
        metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.875}},
                      {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 1}},
                      {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.874}},
                      {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.84375}},
                      {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.90625}}]
        all_metrics = evaluator.evaluateWithSeveralMetrics(dataset, metricSets = metricSets)
        rounded_metrics = [{list(x.items())[0][0]: round(list(x.items())[0][1], 3)} for x in all_metrics]
        expected_metrics = [{"precisionAtGivenRecall at recallValue 0.875": 0.875},
                            {"precisionAtGivenRecall at recallValue 1": 0.444},
                            {"precisionAtGivenRecall at recallValue 0.874": 0.875},
                            {"precisionAtGivenRecall at recallValue 0.84375": 0.867},
                            {"precisionAtGivenRecall at recallValue 0.90625": 0.875}]
        self.assertEqual(rounded_metrics, expected_metrics,
                         "Expected metrics: {0}; Result rounded metrics: {1};".format(expected_metrics, rounded_metrics))
        
if __name__ == "__main__":
    unittest.main()
