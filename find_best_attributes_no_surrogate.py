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

from optimization.ILSPopNoSurrogate import ILSPopSurrogate
from macop.solutions.discrete import BinarySolution
from macop.evaluators.base import Evaluator

from macop.operators.discrete.mutators import SimpleMutation
from macop.operators.discrete.mutators import SimpleBinaryMutation
from macop.operators.discrete.crossovers import SimpleCrossover
from macop.operators.discrete.crossovers import RandomSplitCrossover
from optimization.operators.SimplePopCrossover import SimplePopCrossover, RandomPopCrossover

from macop.policies.reinforcement import UCBPolicy

from macop.callbacks.classicals import BasicCheckpoint
from macop.callbacks.policies import UCBCheckpoint
from optimization.callbacks.MultiPopCheckpoint import MultiPopCheckpoint
from optimization.callbacks.SurrogateMonoCheckpoint import SurrogateMonoCheckpoint
#from sklearn.ensemble import RandomForestClassifier

# variables and parameters
models_list         = cfg.models_names_list

from warnings import simplefilter
simplefilter("ignore")

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
    # scene_name; zone_id; image_index_end; label; data

    dataset_train = pd.read_csv(filename + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(filename + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.iloc[:, 3] == 1]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 3] == 0]
    #nb_noisy_train = len(noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 3] == 1]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 3] == 0]
    #nb_noisy_test = len(noisy_df_test.index)

    # use of all data
    final_df_train = pd.concat([not_noisy_df_train, noisy_df_train])
    final_df_test = pd.concat([not_noisy_df_test, noisy_df_test])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    # use of the whole data set for training
    x_dataset_train = final_df_train.iloc[:, 4:]
    x_dataset_test = final_df_test.iloc[:, 4:]

    y_dataset_train = final_df_train.iloc[:, 3]
    y_dataset_test = final_df_test.iloc[:, 3]

    return x_dataset_train, y_dataset_train, x_dataset_test, y_dataset_test

def _get_best_model(X_train, y_train):

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 5, 10, 100]
    param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

    svc = svm.SVC(probability=True, class_weight='balanced')
    #clf = GridSearchCV(svc, param_grid, cv=5, verbose=1, scoring=my_accuracy_scorer, n_jobs=-1)
    clf = GridSearchCV(svc, param_grid, cv=5, verbose=0, n_jobs=-1)

    clf.fit(X_train, y_train)

    model = clf.best_estimator_

    return model

def main():

    parser = argparse.ArgumentParser(description="Train and find best filters to use for model")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--start_surrogate', type=int, help='number of evalution before starting surrogare model', required=True)
    parser.add_argument('--train_every', type=int, help='max number of evalution before retraining surrogare model', required=True)
    parser.add_argument('--length', type=int, help='max data length (need to be specify for evaluator)', required=True)
    parser.add_argument('--pop', type=int, help='pop size', required=True)
    parser.add_argument('--order', type=int, help='walsh order function', required=True)
    parser.add_argument('--ils', type=int, help='number of total iteration for ils algorithm', required=True)
    parser.add_argument('--ls', type=int, help='number of iteration for Local Search algorithm', required=True)
    parser.add_argument('--output', type=str, help='output surrogate model name')

    args = parser.parse_args()

    p_data_file = args.data
    p_length    = args.length
    p_pop       = args.pop
    p_order     = args.order
    p_start     = args.start_surrogate
    p_retrain   = args.train_every
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
        return BinarySolution.random(p_length, validator)


    class RandomForestEvaluator(Evaluator):

        # define evaluate function here (need of data information)
        def compute(self, solution):
            start = datetime.datetime.now()

            # get indices of filters data to use (filters selection from solution)
            indices = []

            for index, value in enumerate(solution.data): 
                if value == 1: 
                    indices.append(index) 

            # keep only selected filters from solution
            x_train_filters = self._data['x_train'].iloc[:, indices]
            y_train_filters = self._data['y_train']
            x_test_filters = self._data['x_test'].iloc[:, indices]
            
            # model = _get_best_model(x_train_filters, y_train_filters)
            model = RandomForestClassifier(n_estimators=500, class_weight='balanced', bootstrap=True, max_samples=0.75, n_jobs=-1)
            model = model.fit(x_train_filters, y_train_filters)
            
            y_test_model = model.predict(x_test_filters)
            test_roc_auc = roc_auc_score(self._data['y_test'], y_test_model)

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
    surrogate_performanche_file_path = os.path.join(cfg.output_surrogates_data_folder, p_output + '_performance.csv')

    # prepare optimization algorithm (only use of mutation as only ILS are used here, and local search need only local permutation)
    operators = [SimpleBinaryMutation(), SimpleMutation(), RandomPopCrossover(), SimplePopCrossover()]
    policy = UCBPolicy(operators, C=100, exp_rate=0.1)

    # define first line if necessary
    if not os.path.exists(surrogate_output_data):
        with open(surrogate_output_data, 'w') as f:
            f.write('x;y\n')

    # custom ILS for surrogate use
    algo = ILSPopSurrogate(initalizer=init, 
                        evaluator=RandomForestEvaluator(data={'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}), # same evaluator by default, as we will use the surrogate function
                        operators=operators, 
                        policy=policy, 
                        validator=validator,
                        population_size=p_pop,
                        surrogate_file_path=surrogate_output_model,
                        start_train_surrogate=p_start, # start learning and using surrogate after 1000 real evaluation
                        solutions_file=surrogate_output_data,
                        walsh_order=p_order,
                        inter_policy_ls_file=os.path.join(backup_model_folder, p_output + '_ls_ucbPolicy.csv'),
                        ls_train_surrogate=p_retrain,
                        maximise=True)
    
    algo.addCallback(MultiPopCheckpoint(every=1, filepath=backup_file_path))
    algo.addCallback(UCBCheckpoint(every=1, filepath=ucb_backup_file_path))
    algo.addCallback(SurrogateMonoCheckpoint(every=1, filepath=surrogate_performanche_file_path))

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


    line_info = p_output + ';' + p_data_file + ';' + str(bestSol.data) + ';' + str(list(bestSol.data).count(1)) + ';' + str(filters_counter) + ';' + str(bestSol.fitness)

    # check if results are already saved...
    already_saved = False

    if os.path.exists(filename_path):
        with open(filename_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                output_name = line.split(';')[0]
                
                if p_output == output_name:
                    already_saved = True

    if not already_saved:
        with open(filename_path, 'a') as f:
            f.write(line_info + '\n')
    
    print('Result saved into %s' % filename_path)


if __name__ == "__main__":
    main()