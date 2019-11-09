def decisionTreeClassifier(trainingData, testData, ncolumns, schemaNames):
	from pyspark.ml import Pipeline
	from pyspark.ml.classification import DecisionTreeClassifier
	from pyspark.ml.tuning import ParamGridBuilder
	from pyspark.ml.feature import StringIndexer, VectorIndexer
	from pyspark.ml.tuning import CrossValidator
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator
	import numpy as np
	from pyspark.ml.evaluation import BinaryClassificationEvaluator
	import time


	dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=15, maxBins=15, impurity='entropy')
	timer = ''
	start = time.time()
	cvModelDT = dt.fit(trainingData)
	end = time.time()
	timer = ((end - start)/60)


	prediction = cvModelDT.transform(testData)

	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="label", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(prediction)

	# Evaluate model
	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	areaUC = evaluator.evaluate(prediction)

	fi = cvModelDT.featureImportances
	imp_feat = np.zeros(ncolumns-1)
	imp_feat[fi.indices] = fi.values
	x = np.arange(ncolumns-1)
	idx = (-imp_feat).argsort()[:3]
	feat = []
	for i in idx:    
	    feat.append(schemaNames[i])
	return feat, accuracy, areaUC, timer



def LogisticRegression(trainingData, testData, schemaNames):
	from pyspark.ml import Pipeline
	from pyspark.ml.classification import LogisticRegression
	from pyspark.ml.tuning import ParamGridBuilder
	from pyspark.ml.feature import StringIndexer, VectorIndexer
	from pyspark.ml.tuning import CrossValidator
	from pyspark.ml.evaluation import BinaryClassificationEvaluator
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator
	import numpy as np
	import time


	lr = LogisticRegression(featuresCol='features', labelCol='label', regParam=0.1, maxIter=7)
	timer = ''
	start = time.time()
	cvModel = lr.fit(trainingData)
	end = time.time()
	timer = ((end - start)/60)

	prediction = cvModel.transform(testData)
	evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(prediction)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	areaUC = evaluator.evaluate(prediction)

	w_r = cvModel.coefficients
	w_r = w_r.tolist()
	feat = []
	for i in (w_r)[-3:][::-1]:
		feat.append(schemaNames[(w_r.index(i))])
	return feat, accuracy, areaUC, timer

def decisionTreeRegressor(data, ncolumns, schemaNames):
	from pyspark.ml import Pipeline
	from pyspark.ml.regression import DecisionTreeRegressor
	from pyspark.ml.tuning import ParamGridBuilder
	from pyspark.ml.feature import StringIndexer, VectorIndexer
	from pyspark.ml.tuning import CrossValidator
	from pyspark.ml.evaluation import RegressionEvaluator
	from pyspark.ml.feature import Binarizer
	from pyspark.ml.evaluation import BinaryClassificationEvaluator
	import numpy as np
	import time



	binarizer = Binarizer(threshold=0.00001, inputCol="features", outputCol="binarized_features", )
	binarizedDataFrame = binarizer.transform(data)

	(trainingData, testData) = binarizedDataFrame.randomSplit([0.9, 0.1], 50)
	dtr = DecisionTreeRegressor(labelCol="label", featuresCol="binarized_features", maxDepth=10, maxBins=10, impurity='Variance')
	
	timer = ''
	start = time.time()
	cvModel = dtr.fit(trainingData)
	end = time.time()
	timer = ((end - start)/60)


	prediction = cvModel.transform(testData)
	evaluator = RegressionEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(prediction)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	areaUC = evaluator.evaluate(prediction)
	
	fi = cvModel.featureImportances
	imp_feat = np.zeros(ncolumns-1)
	imp_feat[fi.indices] = fi.values
	x = np.arange(ncolumns-1)
	idx = (-imp_feat).argsort()[:3]
	feat = []
	for i in idx:    
	    feat.append(schemaNames[i])

	return feat, rmse, areaUC, timer

def main():
	import time
	import pyspark
	from pyspark.sql import SparkSession
	import numpy as np
	import functools
	spark = SparkSession.builder.master("local[20]").appName("COM6012 Decision Trees Regression").getOrCreate()
	sc = spark.sparkContext
	rawdata = spark.read.csv('/home/acp18sak/.conda/envs/jupyter-spark/Assignment2/Data/HIGGS.csv.gz')
	# rawdata.cache()

	sc.setLogLevel("WARN")

	
	newNames = ('label','lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb')
	oldColumns = rawdata.schema.names
	df = functools.reduce(lambda rawdata, idx: rawdata.withColumnRenamed(oldColumns[idx], newNames[idx]), range(len(oldColumns)), rawdata)

	schemaNames = df.schema.names
	ncolumns = len(df.columns)
	from pyspark.sql.types import DoubleType
	for i in range(ncolumns):
	    df = df.withColumn(schemaNames[i], df[schemaNames[i]].cast(DoubleType()))
	from pyspark.ml.feature import VectorAssembler
	assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns], outputCol = 'features') 
	raw_plus_vector = assembler.transform(df)
	data = raw_plus_vector.select('features','label')

	(trainingData, testData) = data.randomSplit([0.9, 0.1], 50)
	
	feat, accuracy, areaUC, training_time = decisionTreeClassifier(trainingData, testData, ncolumns, schemaNames)
	print('\n\n\n ----------------------------------------------------------------------------')
	print('\t-------------- Results for DecisionTreeClassifier --------------\n')
	print("Training Time in minutes: ", training_time)
	print("\nAccuracy for DecisionTreeClassifier = %g " % (accuracy))
	print("AreaUndertheCurve for DecisionTreeClassifier = %g " % areaUC)
	print('\n Top Three Features for Decision Tree Classifier\n')
	for i in range(len(feat)):
		print(i+1, ' ->' ,feat[i])
	print('\n ----------------------------------------------------------------------------\n\n')

	feat, accuracy, areaUC, training_time = LogisticRegression(trainingData, testData, schemaNames)
	print('\n\n\n ----------------------------------------------------------------------------')
	print('\t-------------- Results for LogisticRegression --------------\n')
	print("Training Time in minutes: ", training_time)
	print("\nAccuracy for LogisticRegression = %g " % (accuracy))
	print("AreaUndertheCurve for LogisticRegression = %g " % areaUC)
	print('\n Top Three Features for LogisticRegression\n')
	for i in range(len(feat)):
		print(i+1, ' ->' ,feat[i])
	print('\n ----------------------------------------------------------------------------\n\n')

	feat, rmse, areaUC, training_time = decisionTreeRegressor(data, ncolumns, schemaNames)
	print('\n\n\n ----------------------------------------------------------------------------')
	print('\t-------------- Results for DecisionTreeRegressor --------------\n')
	print("Training Time in minutes: ", training_time)
	print("\nRMSE for DecisionTreeRegressor = %g " % (rmse))
	print("AreaUndertheCurve for DecisionTreeRegressor = %g " % areaUC)
	print('\n Top Three Features for Decision Tree Regressor\n')
	for i in range(len(feat)):
		print(i+1, ' ->' ,feat[i])

	print('\n ----------------------------------------------------------------------------\n\n')

	spark.close()

if __name__ == '__main__':
    main()

