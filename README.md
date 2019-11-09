# Searching-for-exotic-particles-in-high-energy-physics-using-classic-supervised-learning-algorithms
Supervised classification algorithms employed to explore and identify Higgs bosons from particle collisions, like the ones produced in the ​Large Hadron Collider​. HIGGS dataset is used.​.

About the data: “The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. The last 500,000 examples are used as a test set.”

Decision Trees for Classification, Decision Trees for Regression and Logistic Regression are employed over a subset of the dataset and over the full dataset. As performance measures use classification accuracy and ​area under the curve​. For Regression, use a sensible threshold for binarizing the decision.

1. Used pipelines and cross-validation to find the best configuration of parameters and their accuracy. Also a sensible grid for the parameters (for example, three options for each parameter) is constructed. Used the same splits of training and test data when comparing performances between the algorithms. ​In order to find the best configuration of parameters, 25% of the data chosen randomly from the whole set is used​.
     
2. Once the best parameter configurations for each algorithm in the smaller subset of the data are found, Classifiers are trained and tested on the full dataset to compare the performance of the three algorithms in the cluster. Once again, used the same splits of training and test data when comparing performances between the algorithms. Provide training times when using 10 CORES and 20 CORES on University of Sheffield's Cluster Computing platform, (SHARC). 

3. At the end reported the three most relevant features for classification or regression for each ML Algorithm, obtained in step 2.

The dataset is 2.6gb in size. All the experiments are performed on University of Sheffield's Cluster Computing Platform, SHARC. PySpark was employed to avail parallel computing on this huge, compute intensive task.

This dataset was used in the paper ​“Searching for Exotic Particles in High-energy Physics with Deep Learning” by P. Baldi, P. Sadowski, and D. Whiteson, published in Nature Communications 5 (July 2, 2014).
