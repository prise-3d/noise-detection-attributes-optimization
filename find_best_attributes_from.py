# main imports
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import datetime

# model imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import joblib
import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl

# variables and parameters
models_list         = cfg.models_names_list
number_of_values    = 30
ils_iteration       = 4000
ls_iteration        = 10

# default validator
def validator(solution):

    if list(solution.data).count(1) < 5:
        return False

    return True

def loadDataset(filename):

    ########################
    # 1. Get and prepare data
    ########################
    dataset_train = pd.read_csv(filename + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(filename + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 0]
    #nb_noisy_train = len(noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 0]
    #nb_noisy_test = len(noisy_df_test.index)

    # use of all data
    final_df_train = pd.concat([not_noisy_df_train, noisy_df_train])
    final_df_test = pd.concat([not_noisy_df_test, noisy_df_test])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    # use of the whole data set for training
    x_dataset_train = final_df_train.iloc[:,1:]
    x_dataset_test = final_df_test.iloc[:,1:]

    y_dataset_train = final_df_train.iloc[:,0]
    y_dataset_test = final_df_test.iloc[:,0]

    return x_dataset_train, y_dataset_train, x_dataset_test, y_dataset_test

def main():

    parser = argparse.ArgumentParser(description="Train and find best filters to use for model")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--choice', type=str, help='model choice from list of choices', choices=models_list, required=True)
    parser.add_argument('--selector', type=str, help='kind of model to use for selecting', choices=['svm', 'tree'], default='tree')
    parser.add_argument('--length', type=str, help='max data length (need to be specify for evaluator)', required=True)
    parser.add_argument('--output', type=str, help='output name expected for model results', required=True)

    args = parser.parse_args()

    p_data_file = args.data
    p_choice    = args.choice
    p_selector  = args.selector
    p_length    = args.length
    p_output    = args.output

    print(p_data_file)

    # load data from file
    x_train, y_train, x_test, y_test = loadDataset(p_data_file)

    for i in (np.arange(11) + 5):

        model_to_fit = None
        # use of svm here to fit well model
        if p_selector == 'tree':
            model_to_fit = ExtraTreesClassifier(n_estimators=100)

        elif p_selector == 'svm':
            Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            gammas = [0.001, 0.01, 0.1, 5, 10, 100]
            param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

            svc = svm.SVC(probability=True, class_weight='balanced')
            #clf = GridSearchCV(svc, param_grid, cv=5, verbose=1, scoring=my_accuracy_scorer, n_jobs=-1)
            model_to_fit = GridSearchCV(svc, param_grid, cv=5, verbose=1, scoring='roc_auc', n_jobs=-1)

        model = SelectFromModel(model_to_fit, max_features=i)
        selector = model.fit(x_train, y_train)

        binary_selection = [ 0 if x < selector.threshold_ else 1 for x in selector.estimator_.feature_importances_ ]
        X_train_new = selector.transform(x_train)
        X_test_new = selector.transform(x_test)

        print('Shape for {}, is now {}'.format(i, X_train_new.shape))

        svm_model = mdl.get_trained_model(p_choice, X_train_new, y_train)

        y_test_model = svm_model.predict(X_test_new)
        test_roc_auc = roc_auc_score(y_test, y_test_model)
        
        if not os.path.exists(cfg.output_results_folder):
            os.makedirs(cfg.output_results_folder)

        # save model results into file
        with open(os.path.join(cfg.output_results_folder, p_output), 'a') as f:
            line = str(i) + ';'
            line += str(test_roc_auc) + ';'
            
            for index, b in enumerate(binary_selection):

                line += str(b)
                if index < len(binary_selection) - 1:
                    line += ','

            f.write(line + '\n')

    # create `logs` folder if necessary
    if not os.path.exists(cfg.output_logs_folder):
        os.makedirs(cfg.output_logs_folder)

    logging.basicConfig(format='%(asctime)s %(message)s', filename='data/logs/%s.log' % p_data_file.split('/')[-1], level=logging.DEBUG)

    # init solution (`n` attributes)
    


if __name__ == "__main__":
    main()