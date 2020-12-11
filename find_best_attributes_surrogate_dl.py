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

from keras.layers import Dense, Dropout, LSTM, Embedding, GRU, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

import joblib
import sklearn
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

def build_input(df):
    """Convert dataframe to numpy array input with timesteps as float array
    
    Arguments:
        df: {pd.Dataframe} -- Dataframe input
    
    Returns:
        {np.ndarray} -- input LSTM data as numpy array
    """

    arr = df.to_numpy()

    final_arr = []
    for v in arr:
        v_data = []
        for vv in v:
            #scaled_vv = np.array(vv, 'float') - np.mean(np.array(vv, 'float'))
            #v_data.append(scaled_vv)
            v_data.append(vv)
        
        final_arr.append(v_data)
    
    final_arr = np.array(final_arr, 'float32')            

    return final_arr

# default validator
def validator(solution):

    # at least 5 attributes
    if list(solution._data).count(1) < 5:
        return False

    return True

def create_model(input_shape):
    print ('Creating model...')
    model = Sequential()
    #model.add(Embedding(input_dim = 1000, output_dim = 50, input_length=input_length))
    model.add(LSTM(input_shape=input_shape, units=512, activation='tanh', recurrent_activation='sigmoid', dropout=0.4, return_sequences=True))
    model.add(LSTM(units=128, activation='tanh', recurrent_activation='sigmoid', dropout=0.4, return_sequences=True))
    model.add(LSTM(units=32, activation='tanh', dropout=0.4, recurrent_activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  #metrics=['accuracy', tf.keras.metrics.AUC()])
                  metrics=['accuracy'])

    return model

def loadDataset(filename):

    # TODO : load data using DL RNN 

    ########################
    # 1. Get and prepare data
    ########################
    dataset_train = pd.read_csv(filename + '.train', header=None, sep=';')
    dataset_test = pd.read_csv(filename + '.test', header=None, sep=';')

    # getting weighted class over the whole dataset
    # line is composed of :: [scene_name; zone_id; image_index_end; label; data]
    noisy_df_train = dataset_train[dataset_train.iloc[:, 3] == 1]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 3] == 0]
    nb_noisy_train = len(noisy_df_train.index)
    nb_not_noisy_train = len(not_noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 3] == 1]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 3] == 0]
    nb_noisy_test = len(noisy_df_test.index)
    nb_not_noisy_test = len(not_noisy_df_test.index)

    noisy_samples = nb_noisy_test + nb_noisy_train
    not_noisy_samples = nb_not_noisy_test + nb_not_noisy_train

    total_samples = noisy_samples + not_noisy_samples

    print('noisy', noisy_samples)
    print('not_noisy', not_noisy_samples)
    print('total', total_samples)

    class_weight = {
        0: noisy_samples / float(total_samples),
        1: (not_noisy_samples / float(total_samples)),
    }

    # shuffle data
    final_df_train = sklearn.utils.shuffle(dataset_train)
    final_df_test = sklearn.utils.shuffle(dataset_test)

    # split dataset into X_train, y_train, X_test, y_test
    X_train_all = final_df_train.loc[:, 4:].apply(lambda x: x.astype(str).str.split(' '))
    X_train_all = build_input(X_train_all)
    y_train_all = final_df_train.loc[:, 3].astype('int')

    X_test = final_df_test.loc[:, 4:].apply(lambda x: x.astype(str).str.split(' '))
    X_test = build_input(X_test)
    y_test = final_df_test.loc[:, 3].astype('int')

    input_shape = (X_train_all.shape[1], X_train_all.shape[2])
    print('Training data input shape', input_shape)

    # prepare train and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.3, shuffle=False)

    return X_train, X_val, y_train, y_val, X_test, y_test, class_weight


def main():

    parser = argparse.ArgumentParser(description="Train and find best filters to use for model")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--start_surrogate', type=int, help='number of evalution before starting surrogare model', default=100)
    parser.add_argument('--length', type=int, help='max data length (need to be specify for evaluator)', required=True)
    parser.add_argument('--ils', type=int, help='number of total iteration for ils algorithm', required=True)
    parser.add_argument('--ls', type=int, help='number of iteration for Local Search algorithm', required=True)
    parser.add_argument('--every_ls', type=int, help='number of max iteration for retraining surrogate model', required=True)
    parser.add_argument('--output', type=str, help='output surrogate model name')

    args = parser.parse_args()

    p_data_file = args.data
    p_length    = args.length
    p_start     = args.start_surrogate
    p_ils_iteration = args.ils
    p_ls_iteration  = args.ls
    p_every_ls      = args.every_ls
    p_output = args.output

    print(p_data_file)

    # load data from file
    X_train, X_val, y_train, y_val, X_test, y_test, class_weight = loadDataset(p_data_file)

    # create `logs` folder if necessary
    if not os.path.exists(cfg.output_logs_folder):
        os.makedirs(cfg.output_logs_folder)

    logging.basicConfig(format='%(asctime)s %(message)s', filename='data/logs/{0}.log'.format(p_output), level=logging.DEBUG)

    # init solution (`n` attributes)
    def init():
        return BinarySolution([], p_length).random(validator)

    # define evaluate function here (need of data information)
    def evaluate(solution):

        start = datetime.datetime.now()

        # get indices of filters data to use (filters selection from solution)
        indices = []

        for index, value in enumerate(solution._data): 
            if value == 1: 
                indices.append(index) 

        # keep only selected filters from solution
        x_train_filters = X_train[:, :, indices]
        x_val_filters = X_val[:, :, indices]
        x_test_filters = X_test[:, :, indices]
        
        # model = mdl.get_trained_model(p_choice, x_train_filters, y_train_filters)

        # model = RandomForestClassifier(n_estimators=10)
        input_shape = (x_train_filters.shape[1], x_train_filters.shape[2])
        print('Training data input shape', input_shape)
        model = create_model(input_shape)
        model.summary()

        # model = model.fit(x_train_filters, y_train_filters)

        print("Fitting model with custom class_weight", class_weight)
        history = model.fit(x_train_filters, y_train, batch_size=128, epochs=30, validation_data=(x_val_filters, y_val), verbose=1, shuffle=True, class_weight=class_weight)

        
        y_test_model = model.predict(x_test_filters)
        y_test_predict = [ 1 if x > 0.5 else 0 for x in y_test_model ]
        test_roc_auc = roc_auc_score(y_test, y_test_predict)

        end = datetime.datetime.now()
        del model

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

    # custom ILS for surrogate use
    algo = ILSSurrogate(initalizer=init, 
                        evaluator=evaluate, # same evaluator by defadefaultult, as we will use the surrogate function
                        operators=operators, 
                        policy=policy, 
                        validator=validator,
                        surrogate_file_path=surrogate_output_model,
                        start_train_surrogate=p_start, # start learning and using surrogate after 1000 real evaluation
                        solutions_file=surrogate_output_data,
                        ls_train_surrogate=p_every_ls,
                        maximise=True)
    
    algo.addCallback(BasicCheckpoint(every=1, filepath=backup_file_path))
    algo.addCallback(UCBCheckpoint(every=1, filepath=ucb_backup_file_path))
    algo.addCallback(SurrogateCheckpoint(every=p_ls_iteration, filepath=surrogate_backup_file_path)) # try every LS like this

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