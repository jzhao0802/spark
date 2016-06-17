This function is for Logistic Elastic Net regression in ML module using Spark.elasticNetParam corresponds to alpha and regParam corresponds to lambda
	1. elasticNetParam is in range [0, 1]. 
    	   For alpha = 0, the penalty is an L2 penalty. 
           For alpha = 1, it is an L1 penalty.
	2. regParam is regularization parameter (>= 0).
So if we want to ran Lasso model, then elasticNetParam should be 1.

The following are some paramters need to update:
	lambdastart: this is the start value for lambda
	lambdastop:  this is the end value for lambda
	lambdanum:   total count of lambda sequence
	alphastart:  this is the start value for alpha
	alphastop:   this is the end value for alpha
	alphanum:    total count of lambda sequence
	fold:        Folds for CV
	par:         Number of partitions
	CVCriteria:  Criteria for hyper parameters of CV

