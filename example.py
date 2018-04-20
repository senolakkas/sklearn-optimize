from genetic import GeneticSearchCV
import sklearn.datasets
import numpy as np
import pandas as pd

data = sklearn.datasets.load_iris()

X = data["data"]
y = data["target"]

print(X.shape, y.shape)

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

paramgrid = {"kernel": ["rbf"],
             "C": np.logspace(-9, 9, num=25, base=10),
             "gamma": np.logspace(-9, 9, num=25, base=10)}
print("Size: ", len(paramgrid["kernel"]) * len(paramgrid["C"]) * len(paramgrid["gamma"]))

cv = GeneticSearchCV(estimator=SVC(),
                     params=paramgrid,
                     scoring="accuracy",
                     cv=StratifiedKFold(n_splits=2),
                     verbose=True,
                     population_size=50,
                     gene_mutation_prob=0.10,
                     tournament_size=3,
                     generations_number=10)

print(cv.fit(X, y))

print(cv.best_score_, cv.best_params_)

print(pd.DataFrame(cv.cv_results_).sort_values("mean_test_score", ascending=False).head())
