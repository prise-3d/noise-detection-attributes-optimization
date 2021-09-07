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
from sklearn.feature_selection import SelectFromModel

import joblib
import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl
#from sklearn.ensemble import RandomForestClassifier

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


def train_predict_random_forest(x_train, y_train, x_test, y_test):

    print('Start training Random forest model')
    start = datetime.datetime.now()
    
    # model = _get_best_model(x_train_filters, y_train_filters)
    random_forest_model = RandomForestClassifier(n_estimators=500, class_weight='balanced', bootstrap=True, max_samples=0.75, n_jobs=-1)

    # No need to learn
    random_forest_model = random_forest_model.fit(x_train, y_train)
    
    y_test_model = random_forest_model.predict(x_test)
    test_roc_auc = roc_auc_score(y_test, y_test_model)

    end = datetime.datetime.now()

    diff = end - start

    print("Evaluation took: {}, AUC score found: {}".format(divmod(diff.days * 86400 + diff.seconds, 60), test_roc_auc))

    return random_forest_model


def train_predict_selector(model, x_train, y_train, x_test, y_test):

    start = datetime.datetime.now()

    print("Using Select from model with Random Forest")

    selector = RFECV(estimator=model, min_features_to_select=13, verbose=1, n_jobs=-1)
    selector.fit(x_train, y_train)
    x_train_transformed = selector.transform(x_train)
    x_test_transformed = selector.transform(x_test)

    print('Previous shape:', x_train.shape)
    print('New shape:', x_train_transformed.shape)

    # using specific features
    model = RandomForestClassifier(n_estimators=500, class_weight='balanced', bootstrap=True, max_samples=0.75, n_jobs=-1)
    model = model.fit(x_train_transformed, y_train)

    y_test_model= model.predict(x_test_transformed)
    test_roc_auc = roc_auc_score(y_test, y_test_model)

    end = datetime.datetime.now()

    diff = end - start
    print("Evaluation took: {}, AUC score found: {}".format(divmod(diff.days * 86400 + diff.seconds, 60), test_roc_auc))

def main():

    parser = argparse.ArgumentParser(description="Train and find using all data to use for model")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--output', type=str, help='output surrogate model name')

    args = parser.parse_args()

    p_data_file = args.data
    p_output = args.output

    print(p_data_file)

    # load data from file
    x_train, y_train, x_test, y_test = loadDataset(p_data_file)

    # train classical random forest
    random_forest_model = train_predict_random_forest(x_train, y_train, x_test, y_test)

    # train using select from model
    train_predict_selector(random_forest_model, x_train, y_train, x_test, y_test)




if __name__ == "__main__":
    main()