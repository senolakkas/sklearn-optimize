# sklearn-optimize
Optimize hyperparameters for a support vector machine classifier (SVC) in scikit-learn via genetic algorithm

Usage examples
--------------



Example of usage:

```python
import sklearn.datasets
import numpy as np

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

paramgrid = {"kernel": ["rbf", "sigmoid", "linear"],
             "C": np.logspace(-9, 9, num=25, base=10),
             "gamma": np.logspace(-9, 9, num=25, base=10)}

from genetic import GeneticSearchCV

cv = GeneticSearchCV(estimator=SVC(),
                     params=paramgrid,
                     scoring="accuracy",
                     cv=StratifiedKFold(n_splits=4),
                     verbose=1,
                     population_size=50,
                     gene_mutation_prob=0.10,
                     gene_crossover_prob=0.5,
                     tournament_size=3,
                     generations_number=5,
                     n_jobs=4)
cv.fit(X, y)
```