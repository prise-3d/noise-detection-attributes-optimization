# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import joblib
import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# model imports
import joblib

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module


def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")

def loadDataset(filename, n_step = 20):

    ########################
    # 1. Get and prepare data
    ########################
    # scene_name; zone_id; image_index_end; label; data
    head, folder_data = os.path.split(filename)
    dataset_train = pd.read_csv(os.path.join(filename, folder_data + '.train'), header=None, sep=";")
    dataset_test = pd.read_csv(os.path.join(filename, folder_data + '.test'), header=None, sep=";")

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


def train_model(p_data_file, p_solution):

    x_dataset_train, y_dataset_train, x_dataset_test, y_dataset_test = loadDataset(p_data_file)

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

    print("-------------------------------------------")
    # model = mdl.get_trained_model(p_choice, x_dataset_train, y_dataset_train)
    model = RandomForestClassifier(n_estimators=500, class_weight='balanced', bootstrap=True, max_samples=0.75, n_jobs=-1)
    model.fit(x_dataset_train, y_dataset_train)
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

    return model


def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    # parser.add_argument('--solution', type=str, help='entropy file data with estimated threshold to read and compute')
    parser.add_argument('--data', type=str, help='dataset filename prefiloc (without .train and .test)', required=True)
    # parser.add_argument('--dataset', type=str, help='datasets file to load and predict from')
    parser.add_argument('--solution', type=str, help='Data of solution to specify filters to use')
    parser.add_argument('--output', type=str, help="output folder")

    args = parser.parse_args()

    # p_model      = args.model
    p_data_file  = args.data 
    p_output     = args.output
    p_solution   = list(map(int, args.solution.split(' ')))

    # 2. load model and compile it
    model = train_model(p_data_file, p_solution)

    # begin prediction
    if not os.path.exists(p_output):
        os.makedirs(p_output)

    scene_predictions = {}
    data_lines = []

    dataset_files = os.listdir(p_data_file)

    for filename in dataset_files:
        filename_path = os.path.join(p_data_file, filename)

        with open(filename_path, 'r') as f:
            for line in f.readlines():
                data_lines.append(line)

    nlines = len(data_lines)
    ncounter = 0

    for line in data_lines:
        data = line.split(';')

        scene_name = data[0]
        zone_index = int(data[1])

        if scene_name not in scene_predictions:
            scene_predictions[scene_name] = []

            for _ in range(16):
                scene_predictions[scene_name].append([])

        # prepare input data
        # ToDo check data input
        
        input_data = [ l.replace('\n', '').split(' ') for l in data[4:] ]
        input_data = np.array([x for i, x in enumerate(input_data) if p_solution[i] == 1 ], 'float32').flatten()
        # print(input_data.flatten())
        input_data = np.expand_dims(input_data, axis=0)
                
        prob = model.predict(input_data)[0]

        scene_predictions[scene_name][zone_index].append(prob)

        ncounter += 1
        write_progress(float(ncounter / nlines))


    # 6. save predictions results
    for key, blocks_predictions in scene_predictions.items():

        output_file = os.path.join(p_output, key + '.csv')

        f = open(output_file, 'w')
        for i, data in enumerate(blocks_predictions):
            f.write(key + ';')
            f.write(str(i) + ';')

            for v in data:
                f.write(str(v) + ';')
            
            f.write('\n')
        f.close()


if __name__== "__main__":
    main()
