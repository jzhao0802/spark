This function is for Logistic Lasso regression in ML module using Spark.elasticNetParam corresponds to alpha and regParam corresponds to lambda
regParam is regularization parameter (>= 0) and elasticNetParam should be 1.

The following are some paramters need to update:
	lambdastart: this is the start value for lambda
	lambdastop:  this is the end value for lambda
	lambdanum:   total count of lambda sequence
	fold:        Folds for CV
	par:         Number of partitions
	CVCriteria:  Criteria for hyper parameters of CV

