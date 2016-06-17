1. Code "baggingRF_noGSCV_1loop_v1":
   This is for the bagging Random Forest with no GSCV
   There are 2 files :
      for 1 loop: baggingRF_GSCV_loopsim_v*.py
      for multiple simulations: baggingRF_noGSCV_1loop_v*.py

   The initial setting is:
      ts_prop = 0.2
      num_simu = 5
      ratio = 50
      nIter = int(200/ratio)=4
      numTrees = 30
      numDepth = 5
      seed = 42
      par = 800

   submit scripts are:
     1. front run:
        sudo spark-submit --deploy-mode client --master yarn --num-executors 17 --executor-cores 5 baggingRF_noGSCV_1loop_v1.py 'baggingRF_noGSCV_1loop_v1'
     2. back run:
        sudo spark-submit --deploy-mode cluster --master yarn --num-executors 20 --executor-cores 5 --conf spark.yarn.submit.waitAppCompletion=false baggingRF_noGSCV_1loop_v1.py 'baggingRF_noGSCV_1loop_v1'

2. Code "pipeline_ml_spark_RF_with_partitions_patid_v1": 
   This code could run random forest in spark using pipeline. Hyper parameters and cross validation would be applied. the following are parameters:
        start_tree:  this is the start number of trees
	stop_tree:   this is the stop number of trees
	num_tree:    total count of tree sequence
	start_depth: this is the start value for depth
	stop_depth:   this is the end value for depth
	num_depth:    total count of depth sequence
	fold:        Folds for CV
	par:         Number of partitions
	CVCriteria:  Criteria for hyper parameters of CV

3. Code "pipeline_rf_stratify_cleaned":
   This code is for Random Forest with stratify sampling data. the following are parameters:
	start_tree:         this is the start number of trees
	stop_tree:          this is the stop number of trees
	num_tree:           total count of tree sequence
	start_depth:        this is the start value for depth
	stop_depth:         this is the end value for depth
	num_depth:          total count of depth sequence
	fold:               Folds for CV
	nDesiredPartitions: Number of partitions

