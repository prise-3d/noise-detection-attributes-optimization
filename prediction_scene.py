from sklearn.externals import joblib

import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier

import sys, os, getopt
import json

from modules.utils import config as cfg

output_model_folder = cfg.saved_models_folder

def main():

    # TODO : use of argparse
    
    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python prediction_scene.py --data xxxx.csv --model xxxx.joblib --output xxxx --scene xxxx')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:o:s", ["help=", "data=", "model=", "output=", "scene="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python prediction_scene.py --data xxxx.csv --model xxxx.joblib --output xxxx --scene xxxx')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python prediction_scene.py --data xxxx.csv --model xxxx.joblib --output xxxx --scene xxxx')
            sys.exit()
        elif o in ("-d", "--data"):
            p_data_file = a
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--output"):
            p_output = a
        elif o in ("-s", "--scene"):
            p_scene = a
        else:
            assert False, "unhandled option"

    if '.joblib' in p_model_file:
        kind_model = 'sklearn'
        model_ext = '.joblib'

    if '.json' in p_model_file:
        kind_model = 'keras'
        model_ext = '.json'

    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    dataset = pd.read_csv(p_data_file, header=None, sep=";")

    y_dataset = dataset.ix[:,0]
    x_dataset = dataset.ix[:,1:]

    noisy_dataset = dataset[dataset.ix[:, 0] == 1]
    not_noisy_dataset = dataset[dataset.ix[:, 0] == 0]

    y_noisy_dataset = noisy_dataset.ix[:, 0]
    x_noisy_dataset = noisy_dataset.ix[:, 1:]

    y_not_noisy_dataset = not_noisy_dataset.ix[:, 0]
    x_not_noisy_dataset = not_noisy_dataset.ix[:, 1:]

    if kind_model == 'keras':
        with open(p_model_file, 'r') as f:
            json_model = json.load(f)
            model = model_from_json(json_model)
            model.load_weights(p_model_file.replace('.json', '.h5'))

            model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

        _, vector_size = np.array(x_dataset).shape

        # reshape all data
        x_dataset = np.array(x_dataset).reshape(len(x_dataset), vector_size, 1)
        x_noisy_dataset = np.array(x_noisy_dataset).reshape(len(x_noisy_dataset), vector_size, 1)
        x_not_noisy_dataset = np.array(x_not_noisy_dataset).reshape(len(x_not_noisy_dataset), vector_size, 1)


    if kind_model == 'sklearn':
        model = joblib.load(p_model_file)

    if kind_model == 'keras':
        y_pred = model.predict_classes(x_dataset)
        y_noisy_pred = model.predict_classes(x_noisy_dataset)
        y_not_noisy_pred = model.predict_classes(x_not_noisy_dataset)

    if kind_model == 'sklearn':
        y_pred = model.predict(x_dataset)
        y_noisy_pred = model.predict(x_noisy_dataset)
        y_not_noisy_pred = model.predict(x_not_noisy_dataset)

    accuracy_global = accuracy_score(y_dataset, y_pred)
    accuracy_noisy = accuracy_score(y_noisy_dataset, y_noisy_pred)
    accuracy_not_noisy = accuracy_score(y_not_noisy_dataset, y_not_noisy_pred)

    if(p_scene):
        print(p_scene + " | " + str(accuracy_global) + " | " + str(accuracy_noisy) + " | " + str(accuracy_not_noisy))
    else:
        print(str(accuracy_global) + " \t | " + str(accuracy_noisy) + " \t | " + str(accuracy_not_noisy))

        with open(p_output, 'w') as f:
            f.write("Global accuracy found %s " % str(accuracy_global))
            f.write("Noisy accuracy found %s " % str(accuracy_noisy))
            f.write("Not noisy accuracy found %s " % str(accuracy_not_noisy))
            for prediction in y_pred:
                f.write(str(prediction) + '\n')

if __name__== "__main__":
    main()
