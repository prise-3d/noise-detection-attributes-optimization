from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier

features_selection_list = [
    "variance_threshold",
    "kbest",
    "linearSVC",
    "tree",
    "rfecv"
]

def features_selection_method(name, params, X_train, y_train, problem_size):

    indices = []

    if name == "variance_threshold":
        percent_to_keep = float(params)
        #sel = VarianceThreshold(threshold=(percent_to_keep * (1 - percent_to_keep)))
        sel = VarianceThreshold(threshold=(percent_to_keep))
        sel.fit_transform(X_train)

        indices = sel.get_support(indices=True)

    if name == "kbest":
        k_param = int(float(params) * problem_size) # here it's a percent over the whole dataset
        model = SelectKBest(chi2, k=k_param).fit_transform(X_train, y_train)

        indices = model.get_support(indices=True)

    if name == "linearSVC":
        C_param = float(params)
        lsvc = LinearSVC(C=C_param, penalty="l1", dual=False).fit(X_train, y_train)
        model = SelectFromModel(lsvc, prefit=True)

        indices = model.get_support(indices=True)

    if name == "tree":
        n_estimarors_param = int(params)
        clf = ExtraTreesClassifier(n_estimators=n_estimarors_param)
        clf = clf.fit(X_train, y_train)
        model = SelectFromModel(clf, prefit=True)

        indices = model.get_support(indices=True)

    if name == "rfecv":
        cv_param = int(params)
        # Create the RFE object and compute a cross-validated score
        svc = SVC(kernel="linear")
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(cv_param),
                    scoring='roc_auc')
        rfecv.fit(X_train, y_train)

        indices = rfecv.get_support(indices=True)

    return indices