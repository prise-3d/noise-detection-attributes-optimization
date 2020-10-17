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
import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl

from optimization.ILSSurrogate import ILSSurrogate
from macop.solutions.BinarySolution import BinarySolution

from macop.operators.mutators.SimpleMutation import SimpleMutation
from macop.operators.mutators.SimpleBinaryMutation import SimpleBinaryMutation
from macop.operators.crossovers.SimpleCrossover import SimpleCrossover
from macop.operators.crossovers.RandomSplitCrossover import RandomSplitCrossover

from macop.operators.policies.UCBPolicy import UCBPolicy

from macop.callbacks.BasicCheckpoint import BasicCheckpoint
from macop.callbacks.UCBCheckpoint import UCBCheckpoint

from sklearn.ensemble import RandomForestClassifier

# variables and parameters
models_list         = cfg.models_names_list

# default validator
def validator(solution):

    # at least 5 attributes
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
    parser.add_argument('--choice', type=str, help='model choice from list of choices', choices=models_list, default=models_list[0], required=False)
    parser.add_argument('--start_surrogate', type=int, help='number of evalution before starting surrogare model', default=1000)
    parser.add_argument('--length', type=int, help='max data length (need to be specify for evaluator)', required=True)
    parser.add_argument('--ils', type=int, help='number of total iteration for ils algorithm', required=True)
    parser.add_argument('--ls', type=int, help='number of iteration for Local Search algorithm', required=True)
    parser.add_argument('--output', type=str, help='output surrogate model name')

    args = parser.parse_args()

    p_data_file = args.data
    p_choice    = args.choice
    p_length    = args.length
    p_start     = args.start_surrogate
    p_ils_iteration = args.ils
    p_ls_iteration  = args.ls
    p_output = args.output

    print(p_data_file)

    # load data from file
    x_train, y_train, x_test, y_test = loadDataset(p_data_file)

    # create `logs` folder if necessary
    if not os.path.exists(cfg.output_logs_folder):
        os.makedirs(cfg.output_logs_folder)

    logging.basicConfig(format='%(asctime)s %(message)s', filename='data/logs/{0}.log'.format(p_output), level=logging.DEBUG)

    # init solution (`n` attributes)
    def init():
        return BinarySolution([], p_length
        ).random(validator)

    # define evaluate function here (need of data information)
    def evaluate(solution):

        start = datetime.datetime.now()

        # get indices of filters data to use (filters selection from solution)
        indices = []

        for index, value in enumerate(solution.data): 
            if value == 1: 
                indices.append(index) 

        # keep only selected filters from solution
        x_train_filters = x_train.iloc[:, indices]
        y_train_filters = y_train
        x_test_filters = x_test.iloc[:, indices]
        
        # TODO : use of GPU implementation of SVM
        # model = mdl.get_trained_model(p_choice, x_train_filters, y_train_filters)

        model = RandomForestClassifier(n_estimators=10)
        model = model.fit(x_train_filters, y_train_filters)
        
        y_test_model = model.predict(x_test_filters)
        test_roc_auc = roc_auc_score(y_test, y_test_model)

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

    # prepare optimization algorithm (only use of mutation as only ILS are used here, and local search need only local permutation)
    operators = [SimpleBinaryMutation(), SimpleMutation()]
    policy = UCBPolicy(operators)

    # define first line if necessary
    if not os.path.exists(surrogate_output_data):
        with open(surrogate_output_data) as f:
            f.write('x;y\n')

    # custom ILS for surrogate use
    algo = ILSSurrogate(_initalizer=init, 
                        _evaluator=evaluate, # same evaluator by defadefaultult, as we will use the surrogate function
                        _operators=operators, 
                        _policy=policy, 
                        _validator=validator,
                        _surrogate_file_path=surrogate_output_model,
                        _start_train_surrogate=p_start, # start learning and using surrogate after 1000 real evaluation
                        _solutions_file=surrogate_output_data,
                        _ls_train_surrogate=1,
                        _maximise=True)
    
    algo.addCallback(BasicCheckpoint(_every=1, _filepath=backup_file_path))
    algo.addCallback(UCBCheckpoint(_every=1, _filepath=ucb_backup_file_path))

    bestSol = algo.run(p_ils_iteration, p_ls_iteration)

    # print best solution found
    print("Found ", bestSol)

    # save model information into .csv file
    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)

    filename_path = os.path.join(cfg.results_information_folder, cfg.optimization_attributes_result_filename)

    filters_counter = 0
    # count number of filters
    for index, item in enumerate(bestSol.data):
        if index != 0 and index % 2 == 1:

            # if two attributes are used
            if item == 1 or bestSol.data[index - 1] == 1:
                filters_counter += 1


    line_info = p_data_file + ';' + str(p_ils_iteration) + ';' + str(p_ls_iteration) + ';' + str(bestSol.data) + ';' + str(list(bestSol.data).count(1)) + ';' + str(filters_counter) + ';' + str(bestSol.fitness())
    with open(filename_path, 'a') as f:
        f.write(line_info + '\n')
    
    print('Result saved into %s' % filename_path)


if __name__ == "__main__":
    main()