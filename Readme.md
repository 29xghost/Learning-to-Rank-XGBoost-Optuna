# LEARNING-TO-RANK USING XGBOOST
## OVERVIEW
This project aims to perform ‘learning to rank’(LTR) on a given set of query ids’, there are various algorithms that can be used to solve the problem, here LambdaMART was the preferred algorithm, and XGboost(extreme gradient boosting) was used to implement the same, xgboost uses gradient boosted trees to perform various regression (‘XGBRegressor()’), classification (‘XGBClassifier()’) or ranking tasks (‘XGBRanker()’) and, Optuna, a library which uses a ‘Tree-structured Parzen Estimator’, that repetitively creates models to tune hyperparameters for a model, was used for hyperparameter tuning. To calculate the NDCG score, the provided ‘trec_eval’ tool was used. The main aim is to maximize the NDCG score.
## MODEL DESCRIPTION
An XGBoost ‘XGBRanker()’ model was initialized and tuned to get a reasonable NDCG(normalized distributive cumulative gain) score, which is a measure of how good the position of a document for a certain query id has been predicted, which means, a higher NDCG means a better model. There are many parameters involved in an XGBRanker() model out of which, the following were tuned: <br>
●	‘objective’: This helps us choose how we want our algorithm to process the data, the two most common choices are ‘rank: pairwise’, which uses LambdaMART to select a pair of documents and puts the more relevant one above the other(pairwise ranking) and ‘rank:ndcg’ which also uses LambdaMART but this time, takes the whole list of documents into account and tries to rank them according to relevance(listwise ranking). Both were set in the grid to get the best out of the two methods. <br>
●	‘max_depth’: Since XGboost uses trees as estimators to do the given task, this parameter helps set a maximum height, beyond which, an estimator cannot grow. <br>

●	‘min_child_weight’: This simply means the number of samples in a tree node or the ‘purity level’ of a node, when the number of samples reaches below this parameter, the tree stops further splits and the node becomes a leaf. This helps reduce overfitting. <br>
●	‘subsample and colsample_bytree’: These help prevent overfitting by randomly selecting a trainset and random features to train the model for a particular estimator. <br>

●	‘eval_metric’: This provides an evaluation metric for the validation data, ‘mean average precision(map)’ the preferred choice for ranking tasks, but since the objective of this assignment is to maximize the NDCG score, an ‘NDCG’ metric was also checked later in hyperparameter tuning to check whether it produces better results on the validation set.
 
## FEATURE ENGINEERING:

After reading both the training and test files, the ‘data.describe()’ function gave the
following results.

![Picture 1](https://github.com/29xghost/Learning-to-Rank-XGBoost-Optuna/blob/main/Images/Picture1.jpg)

![Picture 2](https://github.com/29xghost/Learning-to-Rank-XGBoost-Optuna/blob/main/Images/Picture2.jpg)



As seen above, some features in the training dataset have no values at all (Figure 1), so training a model on those parameters would be of no use. Also, in the test dataset (Figure 2), a column ‘NumChildPages’ has no values, but it contains some values in the training dataset so training a model on such a feature may hamper the final predictions.
Taking all this into consideration, these columns were dropped from the training and test dataset.
## EXPERIMENTS
●	Grouping by QueryID:
In a ‘learning to rank’ problem, the dataset we are given has a set of queries where each query has a set of associated documents and those documents have features that define their relevance. So, it is understandable that the documents associated to a certain QueryID should be kept together, which means, grouping by QueryID is necessary.
 
●	Splitting the data:
The first rule of thumb while solving a data science problem has always been splitting the data into train and test sets, where, random data points are chosen as training and a smaller dataset is chosen for validation (train_test_split()). But in these LTR problems, randomizing the QueryID’s won’t be a viable option as it will nullify the ‘Groupby’ done in the previous step, also, there’s always an option to set ‘shuffle=False’ but this may split a group of documents of a query id into two parts which will also hamper the training process. To get rid of such problems, all unique QueryIDs were appended into a list, then shuffled, and then, documents of 80% of these QueryIDs were put up for training and the rest 20% for validation, this was done based on the order of occurrence of a QueryID in the randomized list. This process helped nullify both problems mentioned above.

●	Dropping QueryID, DocID, Labels : Columns like QueryID, DocID and Labels need to be dropped, QueryID and DocID would not help in predictions and passing Labels as parameters would overfit the model, so these need to be dropped.
●	Initial NDCG score:
An initial NDCG score was calculated using a model with no user-set parameters, the goal now is to increase this score by tuning the hyperparameters and performing k- fold cross-validation on the found parameters.

●	Hyperparameter tuning Setup: <br>
○	Library used:‘Optuna’ was used for hyperparameter tuning, this library recursively creates models and picks up a random set of parameters from the user-provided grid of parameters, it keeps track of the best-found combination of parameters in each iteration. <br>
○	How optuna Works:Optuna requires the user to create an objective function where the model and a grid of parameters are defined, then, an optuna ‘space’ is created where the hyperparameter tuning takes place by repetitive calls(trials) to the objective function. The no of trials can be set by the user. <br>
●	Hyperparameter Tuning process:Generally, tuning is done by placing all parameter ranges in a grid and running the tuning process, but here, the tuning process was done in a step-by-step process, that is, tuning only a few parameters in one step, getting the best values out of those, then predicting another parameter but this time, keeping previously found parameter values fixed. For this, a total of 7 objective functions were created. In the first function, a grid for ‘objective’,’min_child_weight’ and ‘max_depth’ was given, rest being set to default values, the best set was found to be:

![Picture 3](https://github.com/29xghost/Learning-to-Rank-XGBoost-Optuna/blob/main/Images/Picture3.jpg)

then, in the next step, to get a more precise value of these parameters, a tighter bound was taken as the grid in the second function, however, the best set didn’t change, so these were taken as final values for these parameters. Now, in the third
 
function, the parameters found above were kept fixed along with a grid for gamma and a tighter bound grid for the same in function four, the value of gamma in both functions didn’t affect the best NDCG, so gamma was not taken as a tuned hyperparameter in the final set, in function five and six, ‘subsample’ and ‘colsample_bytree’ were tuned in the same way. Finally, in the last and final function, the ‘eval_metric’ was tuned. After each objective function iteration, the best set of parameters were stored in a dictionary ‘final_params’. The final set of parameters obtained were:
final_params=  {'eval_metric':'ndcg',  'objective':'rank:pairwise',  'max_depth':3
,'min_child_weight':1, 'subsample':0.8, 'colsample_bytree':1 }. <br>

●	K-Fold Cross-Validation:
Finally, the ‘sklearn.GroupKFold’ library was used to divide the data into 10 folds (ten pairs of training and validation), this library helps avoid the possibility of overlapping groups i.e, the document details of one QueryID would remain together and won’t randomize/split into parts, which is the only way a ‘learning to rank’ dataset should be handled. The best model out of all these folds was saved and used to predict labels for the given test data.
Here’s a plot showing the progress in NDCG scores with objective functions along with the best KFold score:

![Picture 4](https://github.com/29xghost/Learning-to-Rank-XGBoost-Optuna/blob/main/Images/Picture4.jpg)



The test run produced from the given test dataset using these hyperparameters gave an NDCG score of 0.6215. Running the same model with exact same parameters on Linux gave different predictions, both run files have been enclosed in the zip.
## LIBRARIES USED
Key libraries used in this assignment were:

LIBRARY	VERSION
Pandas	1.3.2
Numpy	1.20.3
Scikit-learn	0.24.2
XGBoost	1.4.2
Optuna	2.9.1
 

## REFERENCES
References were taken from official docs of imported libraries, cited below:


Article title:	XGBoost Parameters — xgboost 1.6.0-dev documentation Website title: Xgboost.readthedocs.io
URL:	https://xgboost.readthedocs.io/en/latest/parameter.html

Article title:	Optuna - A hyperparameter optimization framework Website title: Optuna
URL:	https://optuna.org/#code_examples

Article title:	sklearn.model_selection.GroupKFold Website title: scikit-learn
URL:	https://scikit- learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html

# Functionality
## Initial Steps:
 
conda create -n <"envirenment name"> python=3.8 <br>
conda activate <"environment name"> <br>
pip install -r requirements.txt


## Running:

Python file : A2.py <br>
-full_sweep(in the __main__ function) is set to False by default, this will generate a new run file based on pre-trained model predictions <br>
-Setting full_sweep=True, would start hyperparameter tuning,then K-fold cross validation and then make a new run file
	based on a newly tuned model

## Files and Descriptions:


A2 OLD.tsv : best test run out of the three sent for scoring (NDCG 0.6215) <br>
A2.tsv- Run generated by the exact same model on Linux OS. <br>
Final_Tuned_Model.json : Pre-trained model which will be picked up to predict and generate a new 
			tsv if full_sweep=False <br>
test.tsv : given test dataset <br>
	
train.tsv : given train dataset <br>

requirements.txt : libraries required for reproducibility


