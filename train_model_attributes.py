# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# models imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import joblib
import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl

# variables and parameters
saved_models_folder = cfg.output_models
models_list         = cfg.models_names_list

current_dirpath     = os.getcwd()
output_model_folder = os.path.join(current_dirpath, saved_models_folder)

def loadDataset(filename, n_step):

    ########################
    # 1. Get and prepare data
    ########################
    # scene_name; zone_id; image_index_end; label; data

    dataset_train = pd.read_csv(filename + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(filename + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    dataset_train = dataset_train[dataset_train.iloc[:, 2] % n_step == 0]
    dataset_test = dataset_test[dataset_test.iloc[:, 2] % n_step == 0]

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

def main():

    parser = argparse.ArgumentParser(description="Train SKLearn model and save it into .joblib file")

    parser.add_argument('--data', type=str, help='dataset filename prefiloc (without .train and .test)', required=True)
    parser.add_argument('--output', type=str, help='output file name desired for model (without .joblib extension)', required=True)
    parser.add_argument('--choice', type=str, help='model choice from list of choices', choices=models_list, required=True)
    parser.add_argument('--step', type=int, help='step number of samples expected', default=20)
    parser.add_argument('--solution', type=str, help='Data of solution to specify filters to use')

    args = parser.parse_args()

    p_data_file = args.data
    p_output    = args.output
    p_step      = args.step
    p_choice    = args.choice
    p_solution  = list(map(int, args.solution.split(' ')))

    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    ########################
    # 1. Get and prepare data
    ########################
    x_dataset_train, y_dataset_train, x_dataset_test, y_dataset_test = loadDataset(p_data_file, p_step)

    # get indices of filters data to use (filters selection from solution)
    indices = []

    print(p_solution)
    for index, value in enumerate(p_solution): 
        if value == 1: 
            indices.append(index) 

    print(f'Selected indices are: {indices}')
    print(f"Train dataset size {len(x_dataset_train)}")
    print(f"Test dataset size {len(x_dataset_test)}")

    x_dataset_train = x_dataset_train.iloc[:, indices]
    x_dataset_test =  x_dataset_test.iloc[:, indices]

    print()

    #######################
    # 2. Construction of the model : Ensemble model structure
    #######################

    print("-------------------------------------------")
    model = mdl.get_trained_model(p_choice, x_dataset_train, y_dataset_train)

    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################
    val_scores = cross_val_score(model, x_dataset_train, y_dataset_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (val_scores.mean(), val_scores.std() * 2))

    ######################
    # 4. Metrics
    ######################

    y_train_model = model.predict(x_dataset_train)
    y_test_model = model.predict(x_dataset_test)

    train_accuracy = accuracy_score(y_dataset_train, y_train_model)
    test_accuracy = accuracy_score(y_dataset_test, y_test_model)

    train_auc = roc_auc_score(y_dataset_train, y_train_model)
    test_auc = roc_auc_score(y_dataset_test, y_test_model)

    ###################
    # 5. Output : Print and write all information in csv
    ###################

    print("Train dataset size ", len(x_dataset_train))
    print("Train acc: ", train_accuracy)
    print("Train AUC: ", train_auc)
    print("Test dataset size ", len(x_dataset_test))
    print("Test acc: ", test_accuracy)
    print("Test AUC: ", test_auc)

    ##################
    # 6. Save model : create path if not exists
    ##################

    if not os.path.exists(saved_models_folder):
        os.makedirs(saved_models_folder)

    joblib.dump(model, output_model_folder + '/' + p_output + '.joblib')

if __name__== "__main__":
    main()
