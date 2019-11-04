# models imports
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
import sklearn.svm as svm


def _get_best_model(X_train, y_train):

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 5, 10, 100]
    param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy', verbose=0)

    clf.fit(X_train, y_train)

    model = clf.best_estimator_

    return model

def svm_model(X_train, y_train):

    return _get_best_model(X_train, y_train)


def ensemble_model(X_train, y_train):

    svm_model = _get_best_model(X_train, y_train)

    lr_model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=1)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

    ensemble_model = VotingClassifier(estimators=[
       ('svm', svm_model), ('lr', lr_model), ('rf', rf_model)], voting='soft', weights=[1,1,1])

    ensemble_model.fit(X_train, y_train)

    return ensemble_model


def ensemble_model_v2(X_train, y_train):

    svm_model = _get_best_model(X_train, y_train)
    knc_model = KNeighborsClassifier(n_neighbors=2)
    gbc_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    lr_model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=1)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

    ensemble_model = VotingClassifier(estimators=[
       ('lr', lr_model),
       ('knc', knc_model),
       ('gbc', gbc_model),
       ('svm', svm_model),
       ('rf', rf_model)],
       voting='soft', weights=[1, 1, 1, 1, 1])

    ensemble_model.fit(X_train, y_train)

    return ensemble_model

def rfe_svm_model(X_train, y_train, n_components=1):

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 5, 10, 100]
    param_grid = {'estimator__kernel':['rbf'], 'estimator__C': Cs, 'estimator__gamma' : gammas}

    svc = svm.SVC(probability=True)

    rfe_model = RFECV(svc, step=1, cv=10, verbose=0)

    clf = GridSearchCV(rfe_model, param_grid, cv=10, scoring='accuracy', verbose=1)

    clf.fit(X_train, y_train)

    print(clf.best_estimator_)
    print('------------------------------')
    print(clf.best_estimator_.n_features_)
    print('------------------------------')
    print(clf.best_estimator_.ranking_)
    print('------------------------------')
    print(clf.best_estimator_.grid_scores_)

    return clf.best_estimator_.estimator_


def get_trained_model(choice, X_train, y_train):

    if choice == 'svm_model':
        return svm_model(X_train, y_train)

    if choice == 'ensemble_model':
        return ensemble_model(X_train, y_train)

    if choice == 'ensemble_model_v2':
        return ensemble_model_v2(X_train, y_train)

    if choice == 'rfe_svm_model':
        return rfe_svm_model(X_train, y_train)