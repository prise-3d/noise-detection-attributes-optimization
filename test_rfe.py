from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
param_grid = [{'estimator__C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=4)
clf = GridSearchCV(selector, param_grid, cv=7)
clf.fit(X, y)
print(clf.best_estimator_.estimator_)
print(clf.best_estimator_.grid_scores_)
print(clf.best_estimator_.ranking_)