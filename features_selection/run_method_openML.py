import os, argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn.svm as svm

from methods import features_selection_list, features_selection_method


def train_model(X_train, y_train):

    print ('Creating model...')
    # here use of SVM with grid search CV
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1,10, 100, 1000]
    param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

    svc = svm.SVC(probability=True, class_weight='balanced')
    clf = GridSearchCV(svc, param_grid, cv=2, verbose=1, n_jobs=-1)

    clf.fit(X_train, y_train)

    model = clf.best_estimator_

    return model

def loadDataset(filename):

    ########################
    # 1. Get and prepare data
    ########################
    dataset = pd.read_csv(filename, sep=',')

    # change label as common
    min_label_value = min(dataset.iloc[:, -1])
    max_label_value = max(dataset.iloc[:, -1])

    dataset.iloc[:, -1] = dataset.iloc[:, -1].replace(min_label_value, 0)
    dataset.iloc[:, -1] = dataset.iloc[:, -1].replace(max_label_value, 1)

    X_dataset = dataset.iloc[:, :-1]
    y_dataset = dataset.iloc[:, -1]

    problem_size = len(X_dataset.columns)

    # min/max normalisation over feature
    # create a scaler object
    scaler = MinMaxScaler()
    # fit and transform the data
    X_dataset = np.array(pd.DataFrame(scaler.fit_transform(X_dataset), columns=X_dataset.columns))

    # prepare train, validation and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.3, shuffle=True)

    return X_train, y_train, X_test, y_test, problem_size


def main():

    parser = argparse.ArgumentParser(description="Get features extraction from specific method")

    parser.add_argument('--data', type=str, help='open ml dataset filename prefix', required=True)
    parser.add_argument('--method', type=str, help='method name to use', choices=features_selection_list, required=True)
    parser.add_argument('--params', type=str, help='params used for the current selected method', required=True)
    parser.add_argument('--ntrain', type=int, help='number of training in order to keep mean of score', default=1)
    parser.add_argument('--output', type=str, help='output features selection results')

    args = parser.parse_args()

    p_data_file = args.data
    p_method    = args.method
    p_params    = args.params
    p_ntrain    = args.ntrain
    p_output    = args.output

    # load data from file and get problem size
    X_train, y_train, X_test, y_test, problem_size = loadDataset(p_data_file)

    # extract indices selected features
    features_indices = features_selection_method(p_method, p_params, X_train, y_train, problem_size)

    print(f'Selected features {len(features_indices)} over {problem_size}')

    auc_scores = []
    acc_scores = []
    
    for i in range(p_ntrain):

        # new split of dataset
        X_train, y_train, X_test, y_test, problem_size = loadDataset(p_data_file)

        # get reduced dataset
        X_train_reduced = X_train[:, features_indices]
        X_test_reduced = X_test[:, features_indices]


        # get trained model over reduce dataset
        model = train_model(X_train_reduced, y_train)

        # get predicted labels over test dataset
        y_test_model = model.predict(X_test_reduced)
        y_test_predict = [ 1 if x > 0.5 else 0 for x in y_test_model ]
        test_roc_auc = roc_auc_score(y_test, y_test_predict)
        test_acc = accuracy_score(y_test, y_test_predict)

        print(f'Run n°{i}: {test_roc_auc} (AUC ROC)')

        # append score into list of run
        auc_scores.append(test_roc_auc)
        acc_scores.append(test_acc)

    mean_auc_score = sum(auc_scores) / len(auc_scores)
    mean_acc_score = sum(acc_scores) / len(acc_scores)

    var_acc_score = np.var(acc_scores)
    var_auc_score = np.var(auc_scores)

    std_acc_score = np.std(acc_scores)
    std_auc_score = np.std(auc_scores)

    print(f'Model performance using {p_method} (params: {p_params}) is of {mean_auc_score:.2f}')

    # now save trained model and params obtained
    header_line = 'dataset;method;params;ntrain;n_features;acc_test;auc_test;var_acc_test;var_auc_test;std_acc_test;std_auc_test;features_indices\n'
    data_line = f'{p_data_file};{p_method};{p_params};{p_ntrain};{len(features_indices)};{mean_acc_score};{mean_auc_score};{var_acc_score};{var_auc_score};{std_acc_score};{std_auc_score};{" ".join(list(map(str, features_indices)))}\n'

    output_folder, _ = os.path.split(p_output)

    if len(output_folder) > 0:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    if not os.path.exists(p_output):
        with open(p_output, 'w') as f:
            f.write(header_line)

    with open(p_output, 'a') as f:
        f.write(data_line)
    

if __name__ == "__main__":
    main()