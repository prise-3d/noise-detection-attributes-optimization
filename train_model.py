# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# models imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import sklearn.svm as svm
from sklearn.utils import shuffle
#from sklearn.externals import joblib
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl

# variables and parameters
saved_models_folder = cfg.saved_models_folder
models_list         = cfg.models_names_list

current_dirpath     = os.getcwd()
output_model_folder = os.path.join(current_dirpath, saved_models_folder)


def main():

    parser = argparse.ArgumentParser(description="Train SKLearn model and save it into .joblib file")

    parser.add_argument('--data', type=str, help='dataset filename prefiloc (without .train and .test)')
    parser.add_argument('--output', type=str, help='output file name desired for model (without .joblib extension)')
    parser.add_argument('--choice', type=str, help='model choice from list of choices', choices=models_list)

    args = parser.parse_args()

    p_data_file = args.data
    p_output    = args.output
    p_choice    = args.choice

    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    ########################
    # 1. Get and prepare data
    ########################
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 0]
    nb_noisy_test = len(noisy_df_test.index)

    final_df_train = pd.concat([not_noisy_df_train[0:nb_noisy_train], noisy_df_train])
    final_df_test = pd.concat([not_noisy_df_test[0:nb_noisy_test], noisy_df_test])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    final_df_train_size = len(final_df_train.index)
    final_df_test_size = len(final_df_test.index)

    # use of the whole data set for training
    x_dataset_train = final_df_train.iloc[:,1:]
    x_dataset_test = final_df_test.iloc[:,1:]

    y_dataset_train = final_df_train.iloc[:,0]
    y_dataset_test = final_df_test.iloc[:,0]

    #######################
    # 2. Construction of the model : Ensemble model structure
    #######################

    print("-------------------------------------------")
    print("Train dataset size: ", final_df_train_size)

    model = mdl.get_trained_model(p_choice, x_dataset_train, y_dataset_train)

    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################
    val_scores = cross_val_score(model, x_dataset_train, y_dataset_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (val_scores.mean(), val_scores.std() * 2))

    ######################
    # 4. Test : Validation and test dataset from .test dataset
    ######################

    # we need to specify validation size to 20% of whole dataset
    val_set_size = int(final_df_train_size/3)
    test_set_size = val_set_size

    total_validation_size = val_set_size + test_set_size

    if final_df_test_size > total_validation_size:
        x_dataset_test = x_dataset_test[0:total_validation_size]
        y_dataset_test = y_dataset_test[0:total_validation_size]

    X_test, X_val, y_test, y_val = train_test_split(x_dataset_test, y_dataset_test, test_size=0.2, random_state=1)

    y_test_model = model.predict(X_test)
    y_val_model = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_model)
    test_accuracy = accuracy_score(y_test, y_test_model)

    print('Train dataset 1 ', np.any(y_test_model == 1))
    print('Train dataset 0 ', np.any(y_test_model == 0))

    print('Val dataset 1 ', np.any(y_val_model == 1))
    print('Val dataset 0 ', np.any(y_val_model == 0))

    val_f1 = f1_score(y_val, y_val_model)
    test_f1 = f1_score(y_test, y_test_model)

    ###################
    # 5. Output : Print and write all information in csv
    ###################

    print("Validation dataset size ", val_set_size)
    print("Validation: ", val_accuracy)
    print("Validation F1: ", val_f1)
    print("Test dataset size ", test_set_size)
    print("Test: ", test_accuracy)
    print("Test F1: ", test_f1)

    ##################
    # 6. Save model : create path if not exists
    ##################

    if not os.path.exists(saved_models_folder):
        os.makedirs(saved_models_folder)

    joblib.dump(model, output_model_folder + '/' + p_output + '.joblib')

if __name__== "__main__":
    main()
