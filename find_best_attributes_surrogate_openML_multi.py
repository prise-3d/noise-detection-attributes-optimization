# main imports
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import datetime
import random

# model imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import joblib
import sklearn
import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl

from optimization.ILSMultiSurrogate import ILSMultiSurrogate
from macop.solutions.BinarySolution import BinarySolution

from macop.operators.mutators.SimpleMutation import SimpleMutation
from macop.operators.mutators.SimpleBinaryMutation import SimpleBinaryMutation
from macop.operators.crossovers.SimpleCrossover import SimpleCrossover
from macop.operators.crossovers.RandomSplitCrossover import RandomSplitCrossover

from macop.operators.policies.UCBPolicy import UCBPolicy

from macop.callbacks.BasicCheckpoint import BasicCheckpoint
from macop.callbacks.UCBCheckpoint import UCBCheckpoint
from optimization.callbacks.SurrogateCheckpoint import SurrogateCheckpoint
from optimization.callbacks.MultiSurrogateCheckpoint import MultiSurrogateCheckpoint

from sklearn.ensemble import RandomForestClassifier

# avoid display of warning
def warn(*args, **kwargs):
    pass

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.warn = warn

# default validator
def validator(solution):

    # at least 5 attributes
    if list(solution._data).count(1) < 2:
        return False

    return True

def train_model(X_train, y_train):

    #print ('Creating model...')
    # here use of SVM with grid search CV
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1,10, 100]
    param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

    svc = svm.SVC(probability=True, class_weight='balanced')
    #clf = GridSearchCV(svc, param_grid, cv=5, verbose=1, scoring=my_accuracy_scorer, n_jobs=-1)
    clf = GridSearchCV(svc, param_grid, cv=4, verbose=0, n_jobs=-1)

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

    parser = argparse.ArgumentParser(description="Train and find best filters to use for model")

    parser.add_argument('--data', type=str, help='open ml dataset filename prefix', required=True)
    parser.add_argument('--every_ls', type=int, help='train every ls surrogate model', default=50) # default value
    parser.add_argument('--k_division', type=int, help='number of expected sub surrogate model', default=20)
    parser.add_argument('--k_dynamic', type=int, help='specify if indices for each sub surrogate model are changed or not for each training', default=0, choices=[0, 1])
    parser.add_argument('--k_random', type=int, help='specify if split is random or not', default=1, choices=[0, 1])
    parser.add_argument('--ils', type=int, help='number of total iteration for ils algorithm', required=True)
    parser.add_argument('--ls', type=int, help='number of iteration for Local Search algorithm', required=True)
    parser.add_argument('--generate_only', type=int, help='number of iteration for Local Search algorithm', default=0, choices=[0, 1])
    parser.add_argument('--output', type=str, help='output surrogate model name')

    args = parser.parse_args()

    p_data_file = args.data
    p_every_ls   = args.every_ls
    p_k_division = args.k_division
    p_k_dynamic = bool(args.k_dynamic)
    p_k_random = bool(args.k_random)
    p_ils_iteration = args.ils
    p_ls_iteration  = args.ls
    p_generate_only = bool(args.generate_only)
    p_output = args.output

    # load data from file and get problem size
    X_train, y_train, X_test, y_test, problem_size = loadDataset(p_data_file)

    # create `logs` folder if necessary
    if not os.path.exists(cfg.output_logs_folder):
        os.makedirs(cfg.output_logs_folder)

    logging.basicConfig(format='%(asctime)s %(message)s', filename='data/logs/{0}.log'.format(p_output), level=logging.DEBUG)

    # init solution (`n` attributes)
    def init():
        return BinarySolution([], problem_size).random(validator)

    # define evaluate function here (need of data information)
    def evaluate(solution):

        start = datetime.datetime.now()

        # get indices of filters data to use (filters selection from solution)
        indices = []

        for index, value in enumerate(solution._data): 
            if value == 1: 
                indices.append(index) 

        print(f'Training SVM with {len(indices)} from {len(solution._data)} available features')

        # keep only selected filters from solution
        x_train_filters = X_train[:, indices]
        x_test_filters = X_test[ :, indices]
        
        # model = mdl.get_trained_model(p_choice, x_train_filters, y_train_filters)
        model = train_model(x_train_filters, y_train)

        y_test_model = model.predict(x_test_filters)
        y_test_predict = [ 1 if x > 0.5 else 0 for x in y_test_model ]
        test_roc_auc = roc_auc_score(y_test, y_test_predict)

        end = datetime.datetime.now()

        diff = end - start

        print("Real evaluation took: {}, score found: {}".format(divmod(diff.days * 86400 + diff.seconds, 60), test_roc_auc))

        return test_roc_auc


    # build all output folder and files based on `output` name
    backup_model_folder = os.path.join(cfg.output_backup_folder, p_output)
    surrogate_output_model = os.path.join(cfg.output_surrogates_model_folder, p_output)
    surrogate_output_data = os.path.join(cfg.output_surrogates_data_folder, p_output)

    if not os.path.exists(backup_model_folder):
        os.makedirs(backup_model_folder)

    if not os.path.exists(cfg.output_surrogates_model_folder):
        os.makedirs(cfg.output_surrogates_model_folder)

    if not os.path.exists(cfg.output_surrogates_data_folder):
        os.makedirs(cfg.output_surrogates_data_folder)

    backup_file_path = os.path.join(backup_model_folder, p_output + '.csv')
    ucb_backup_file_path = os.path.join(backup_model_folder, p_output + '_ucbPolicy.csv')
    surrogate_backup_file_path = os.path.join(cfg.output_surrogates_data_folder, p_output + '_train.csv')
    surrogate_k_indices_backup_file_path = os.path.join(cfg.output_surrogates_data_folder, p_output + '_k_indices.csv')

    # prepare optimization algorithm (only use of mutation as only ILS are used here, and local search need only local permutation)
    operators = [SimpleBinaryMutation(), SimpleMutation()]
    policy = UCBPolicy(operators)

    # define first line if necessary
    if not os.path.exists(surrogate_output_data):
        folder, _ = os.path.split(surrogate_output_data)

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(surrogate_output_data, 'w') as f:
            f.write('x;y\n')


    # custom start surrogate variable based on problem size
    p_start = int(0.5 * problem_size)

    # fixed minimal number of real evaluations
    if p_start < 50:
        p_start = 50

    print(f'Starting using surrogate after {p_start} reals training')

    # custom ILS for surrogate use
    algo = ILSMultiSurrogate(initalizer=init, 
                        evaluator=evaluate, # same evaluator by defadefaultult, as we will use the surrogate function
                        operators=operators, 
                        policy=policy, 
                        validator=validator,
                        output_log_surrogates=os.path.join(cfg.output_surrogates_data_folder, 'logs', p_output),
                        surrogates_file_path=surrogate_output_model,
                        start_train_surrogates=p_start, # start learning and using surrogate after 1000 real evaluation
                        solutions_file=surrogate_output_data,
                        ls_train_surrogates=p_every_ls, # retrain surrogate every `x` iteration
                        k_division=p_k_division,
                        k_dynamic=p_k_dynamic,
                        k_random=p_k_random,
                        generate_only=p_generate_only,
                        maximise=True)
    
    algo.addCallback(BasicCheckpoint(every=1, filepath=backup_file_path))
    algo.addCallback(UCBCheckpoint(every=1, filepath=ucb_backup_file_path))
    algo.addCallback(SurrogateCheckpoint(every=p_ls_iteration, filepath=surrogate_backup_file_path)) # try every LS like this
    algo.addCallback(MultiSurrogateCheckpoint(every=p_ls_iteration, filepath=surrogate_k_indices_backup_file_path)) # try every LS like this

    bestSol = algo.run(p_ils_iteration, p_ls_iteration)

    # print best solution found
    print("Found ", bestSol)

    # save model information into .csv file
    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)

    filename_path = os.path.join(cfg.results_information_folder, cfg.optimization_attributes_result_filename)

    line_info = p_data_file + ';' + str(p_ils_iteration) + ';' + str(p_ls_iteration) + ';' + str(bestSol._data) + ';' + str(list(bestSol._data).count(1)) + ';' + str(bestSol.fitness())
    with open(filename_path, 'a') as f:
        f.write(line_info + '\n')
    
    print('Result saved into %s' % filename_path)


if __name__ == "__main__":
    main()