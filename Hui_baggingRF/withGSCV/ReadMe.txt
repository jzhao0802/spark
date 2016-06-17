#This is for the bagging Random Forest with Grid-search and Cross-Validation

#There are 2 files :
#for 1 loop: baggingRF_GSCV_1loop_v*.py
#for multiple simulations: baggingRF_GSCV_loopsim_v*.py

#The initial setting is:
#ts_prop = 0.2               #proportion of test set
#num_simu = 5                #number of simulation, will not be used in 1 loop
#ratio = 50                  #incidence in each bagging random forest model
#numtree = [200, 300]        #number of trees, around 300 
#numdepth = [3, 4]           #number of depth, around 5
#nodesize = [3, 5]           #node size, around 5
#mtry = ['onethird', 'sqrt'] #mtry, select in ('auto', 'all, 'onethird', 'sqrt', 'log2')
#fold = 2                    #number of fold, 3 or 5
#par = 400                   #number of partition, usually 400, 800
#seed = 42                   #seed in random forest model

#submit scripts: 
#1. front run:
#	1 loop:
#	sudo spark-submit --deploy-mode client --master yarn --num-executors 17 --executor-cores 5 --executor-memory 5g --packages com.databricks:spark-csv_2.11:1.3.0  baggingRF_GSCV_1loop_v*.py 'app_name'

#	multi-simulations
#	sudo spark-submit --deploy-mode client --master yarn --num-executors 17 --executor-cores 5 --executor-memory 5g --packages com.databricks:spark-csv_2.11:1.3.0  baggingRF_GSCV_loopsim_v*.py 'app_name'

#2. back run:
#	1 loop:
#	sudo spark-submit --deploy-mode cluster --master yarn --num-executors 17 --executor-cores 5 --executor-memory 5g --packages com.databricks:spark-csv_2.11:1.3.0 --conf spark.yarn.submit.waitAppCompletion=false baggingRF_GSCV_1loop_v*.py 'app_name'

#	multi-simulations
#	sudo spark-submit --deploy-mode cluster --master yarn --num-executors 17 --executor-cores 5 --executor-memory 5g --packages com.databricks:spark-csv_2.11:1.3.0 --conf spark.yarn.submit.waitAppCompletion=false baggingRF_GSCV_loopsim_v*.py 'app_name'

#Note:
#1. The applications are using spark-csv package version 1.3.0 to read in CSV files
#2. change the app_name in the last argument before submitting
#3. cd to the folder where saved the .py file 
#4. options are:
#	--num-executors 17
#	--executor-cores 5
#	--executor-memory 5g

