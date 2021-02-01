# main imports
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging

# model imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl

from macop.macop.algorithms.mono.IteratedLocalSearch import IteratedLocalSearch as ILS
from macop.macop.solutions.BinarySolution import BinarySolution

from macop.macop.operators.mutators.SimpleMutation import SimpleMutation
from macop.macop.operators.mutators.SimpleBinaryMutation import SimpleBinaryMutation
from macop.macop.operators.crossovers.SimpleCrossover import SimpleCrossover
from macop.macop.operators.crossovers.RandomSplitCrossover import RandomSplitCrossover

from macop.macop.operators.policies.UCBPolicy import UCBPolicy

from macop.macop.callbacks.BasicCheckpoint import BasicCheckpoint
from macop.macop.callbacks.UCBCheckpoint import UCBCheckpoint

# variables and parameters
models_list         = cfg.models_names_list
number_of_values    = 26
ils_iteration       = 10000
ls_iteration        = 20

# default validator
def validator(solution):

    if list(solution.data).count(1) < 5:
        return False

    return True

# init solution (13 filters)
def init():
    return BinarySolution([], 13).random(validator)

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

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)')
    parser.add_argument('--choice', type=str, help='model choice from list of choices', choices=models_list)

    args = parser.parse_args()

    p_data_file = args.data
    p_choice    = args.choice

    # load data from file
    x_train, y_train, x_test, y_test = loadDataset(p_data_file)

    # create `logs` folder if necessary
    if not os.path.exists(cfg.output_logs_folder):
        os.makedirs(cfg.output_logs_folder)

    logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/%s.log' % p_data_file.split('/')[-1], level=logging.DEBUG)

    # define evaluate function here (need of data information)
    def evaluate(solution):

        # get indices of filters data to use (filters selection from solution)
        indices = []

        for index, value in enumerate(solution.data): 
            if value == 1: 
                indices.append(index*2) 
                indices.append(index*2+1) 

        # keep only selected filters from solution
        x_train_filters = x_train.iloc[:, indices]
        y_train_filters = y_train
        x_test_filters = x_test.iloc[:, indices]

        model = mdl.get_trained_model(p_choice, x_train_filters, y_train_filters)
        
        y_test_model = model.predict(x_test_filters)
        test_roc_auc = roc_auc_score(y_test, y_test_model)

        return test_roc_auc

    if not os.path.exists(cfg.output_backup_folder):
        os.makedirs(cfg.output_backup_folder)

    backup_file_path = os.path.join(cfg.output_backup_folder, p_data_file.split('/')[-1] + '.csv')
    ucb_backup_file_path = os.path.join(cfg.output_backup_folder, p_data_file.split('/')[-1] + '_ucbPolicy.csv')

    # prepare optimization algorithm
    operators = [SimpleBinaryMutation(), SimpleMutation(), SimpleCrossover(), RandomSplitCrossover()]
    policy = UCBPolicy(operators)

    algo = ILS(init, evaluate, operators, policy, validator, True)
    algo.addCallback(BasicCheckpoint(_every=1, _filepath=backup_file_path))
    algo.addCallback(UCBCheckpoint(_every=1, _filepath=ucb_backup_file_path))

    bestSol = algo.run(ils_iteration, ls_iteration)

    # print best solution found
    print("Found ", bestSol)

    # save model information into .csv file
    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)

    filename_path = os.path.join(cfg.results_information_folder, cfg.optimization_filters_result_filename)

    line_info = p_data_file + ';' + str(ils_iteration) + ';' + str(ls_iteration) + ';' + str(bestSol.data) + ';' + str(list(bestSol.data).count(1)) + ';' + str(bestSol.fitness)
    with open(filename_path, 'a') as f:
        f.write(line_info + '\n')
    
    print('Result saved into %s' % filename_path)


if __name__ == "__main__":
    main()