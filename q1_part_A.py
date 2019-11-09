def decisionTreeClassifier(trainingData, testData, ncolumns, schemaNames):
	from pyspark.ml import Pipeline
	from pyspark.ml.classification import DecisionTreeClassifier
	from pyspark.ml.tuning import ParamGridBuilder
	from pyspark.ml.feature import StringIndexer, VectorIndexer
	from pyspark.ml.tuning import CrossValidator
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator
	import numpy as np
	from pyspark.ml.evaluation import BinaryClassificationEvaluator

	dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
	paramGrid = ParamGridBuilder()\
	    .addGrid(dt.maxDepth, [5, 10, 15]) \
	    .addGrid(dt.maxBins, [5, 10, 15]) \
	    .addGrid(dt.impurity, ['gini','entropy'])\
	    .build()
	pipeline = Pipeline(stages=[dt])
	crossvalDT = CrossValidator(estimator=pipeline,
	                          estimatorParamMaps=paramGrid,
	                          evaluator=MulticlassClassificationEvaluator(metricName='accuracy'),
	                          numFolds=5)
	best_params = ''
	cvModelDT = crossvalDT.fit(trainingData)
	prediction = cvModelDT.transform(testData)

	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="label", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(prediction)

	# Evaluate model
	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	areaUC = evaluator.evaluate(prediction)

	bModel = cvModelDT.bestModel.stages[0]
	fi = bModel.featureImportances

	imp_feat = np.zeros(ncolumns-1)
	imp_feat[fi.indices] = fi.values
	x = np.arange(ncolumns-1)
	idx = (-imp_feat).argsort()[:3]
	feat = []
	for i in idx:    
	    feat.append(schemaNames[i])

	best_params = (cvModelDT.getEstimatorParamMaps()[ np.argmax(cvModelDT.avgMetrics) ])
	return feat, accuracy, areaUC, best_params



def LogisticRegression(trainingData, testData):
	from pyspark.ml import Pipeline
	from pyspark.ml.classification import LogisticRegression
	from pyspark.ml.tuning import ParamGridBuilder
	from pyspark.ml.feature import StringIndexer, VectorIndexer
	from pyspark.ml.tuning import CrossValidator
	from pyspark.ml.evaluation import BinaryClassificationEvaluator
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator
	import numpy as np

	lr = LogisticRegression(featuresCol='features', labelCol='label')
	paramGrid = ParamGridBuilder()\
	    .addGrid(lr.regParam, [0.1, 0.5, 1.0]) \
	    .addGrid(lr.maxIter, [2, 5, 7]) \
	    .build()
	pipeline = Pipeline(stages=[lr])
	crossval = CrossValidator(estimator=pipeline,
	                          estimatorParamMaps=paramGrid,
	                          evaluator=MulticlassClassificationEvaluator(metricName='accuracy'),
	                          numFolds=5)
	best_params = ''
	cvModel = crossval.fit(trainingData)
	prediction = cvModel.transform(testData)

	evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(prediction)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	areaUC = evaluator.evaluate(prediction)
	best_params = (cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])

	return accuracy, areaUC, best_params

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


	binarizer = Binarizer(threshold=0.00001, inputCol="features", outputCol="binarized_features")
	binarizedDataFrame = binarizer.transform(data)

	(trainingData, testData) = binarizedDataFrame.randomSplit([0.8, 0.2], 50)
	dtr = DecisionTreeRegressor(labelCol="label", featuresCol="binarized_features")
	paramGrid = ParamGridBuilder()\
	    .addGrid(dtr.maxDepth, [2, 5, 10]) \
	    .addGrid(dtr.maxBins, [5, 10, 15]) \
	    .addGrid(dtr.impurity, ['Variance'])\
	    .build()
	pipeline = Pipeline(stages=[dtr])
	crossval = CrossValidator(estimator=pipeline,
	                          estimatorParamMaps=paramGrid,
	                          evaluator=RegressionEvaluator(metricName='rmse'),
	                          numFolds=5)
	best_params = ''
	cvModel = crossval.fit(trainingData)
	prediction = cvModel.transform(testData)
	evaluator = RegressionEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(prediction)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	areaUC = evaluator.evaluate(prediction)
	
	bModel = cvModel.bestModel.stages[0]
	fi = bModel.featureImportances
	imp_feat = np.zeros(ncolumns-1)
	imp_feat[fi.indices] = fi.values
	x = np.arange(ncolumns-1)
	idx = (-imp_feat).argsort()[:3]
	feat = []
	for i in idx:    
	    feat.append(schemaNames[i])

	best_params = (cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])

	return feat, rmse, areaUC, best_params

def main():

	import pyspark
	from pyspark.sql import SparkSession
	import numpy as np
	import functools
	spark = SparkSession.builder.master("local[8]").appName("COM6012 Decision Trees Regression").getOrCreate()
	sc = spark.sparkContext
	rawdata = spark.read.csv('/home/acp18sak/.conda/envs/jupyter-spark/Assignment2/Data/higgs_25.csv.gz')
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

	(trainingData, testData) = data.randomSplit([0.8, 0.2], 50)
	
	feat, accuracy, areaUC, best_params = decisionTreeClassifier(trainingData, testData, ncolumns, schemaNames)
	print('\n\n\n ----------------------------------------------------------------------------')
	print('\t-------------- Results for DecisionTreeClassifier --------------\n')
	print("Accuracy for DecisionTreeClassifier = %g " % (accuracy))
	print("AreaUndertheCurve for DecisionTreeClassifier = %g " % areaUC)
	print('\n Top Three Features for Decision Tree Classifier\n')
	for i in range(len(feat)):
		print(i+1, ' ->' ,feat[i])

	print("\nBest Params for DTC are: ", best_params)
	print('\n ----------------------------------------------------------------------------\n\n')

	accuracy, areaUC, best_params = LogisticRegression(trainingData, testData)
	print('\n\n\n ----------------------------------------------------------------------------')
	print('\t-------------- Results for LogisticRegression --------------\n')
	print("Accuracy for LogisticRegression = %g " % (accuracy))
	print("AreaUndertheCurve for LogisticRegression = %g " % areaUC)
	print("\nBest Params for LR are: ", best_params)

	print('\n ----------------------------------------------------------------------------\n\n')

	feat, rmse, areaUC, best_params = decisionTreeRegressor(data, ncolumns, schemaNames)
	print('\n\n\n ----------------------------------------------------------------------------')
	print('\t-------------- Results for DecisionTreeRegressor --------------\n')
	print("RMSE for DecisionTreeRegressor = %g " % (rmse))
	print("AreaUndertheCurve for DecisionTreeRegressor = %g " % areaUC)
	print('\n Top Three Features for Decision Tree Regressor\n')
	for i in range(len(feat)):
		print(i+1, ' ->' ,feat[i])

	print("\nBest Params for DTR are: ", best_params)
	print('\n ----------------------------------------------------------------------------\n\n')

	spark.close()

if __name__ == '__main__':
    main()

