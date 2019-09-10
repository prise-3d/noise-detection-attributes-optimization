# main imports
import sys, os, argparse, json
import numpy as np

# models imports
from keras.models import model_from_json
from sklearn.externals import joblib

# image processing imports
from ipfml import processing, utils
from PIL import Image

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from data_attributes import get_image_features

# variables and parameters
path                  = cfg.dataset_path
min_max_ext           = cfg.min_max_filename_extension
features_choices      = cfg.features_choices_labels
normalization_choices = cfg.normalization_choices

custom_min_max_folder = cfg.min_max_custom_folder

def main():

    # getting all params
    parser = argparse.ArgumentParser(description="Script which detects if an image is noisy or not using specific model")

    parser.add_argument('--image', type=str, help='Image path')
    parser.add_argument('--solution', type=str, help='Data of solution to specify filters to use')
    parser.add_argument('--model', type=str, help='.joblib or .json file (sklearn or keras model)')
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=normalization_choices)
    parser.add_argument('--feature', type=str, help='feature data choice', choices=features_choices)
    parser.add_argument('--custom', type=str, help='Name of custom min max file if use of renormalization of data', default=False)

    args = parser.parse_args()

    p_img_file   = args.image
    p_model_file = args.model
    p_solution   = list(map(int, args.solution.split(' ')))
    p_mode       = args.mode
    p_feature    = args.feature
    p_custom     = args.custom

    if '.joblib' in p_model_file:
        kind_model = 'sklearn'

    if '.json' in p_model_file:
        kind_model = 'keras'

    if kind_model == 'sklearn':
        # load of model file
        model = joblib.load(p_model_file)

    if kind_model == 'keras':
        with open(p_model_file, 'r') as f:
            json_model = json.load(f)
            model = model_from_json(json_model)
            model.load_weights(p_model_file.replace('.json', '.h5'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        features=['accuracy'])

    # load image
    img = Image.open(p_img_file)

    data = get_image_features(p_feature, img)

    # get indices of filters data to use (filters selection from solution)
    indices = []

    for index, value in enumerate(p_solution): 
        if value == 1: 
            indices.append(index*2) 
            indices.append(index*2+1) 

    # check if custom min max file is used
    if p_custom:
        
        test_data = data[indices]
        
        if p_mode == 'svdne':

            # set min_max_filename if custom use
            min_max_file_path = os.path.join(custom_min_max_folder, p_custom)

            # need to read min_max_file
            with open(min_max_file_path, 'r') as f:
                min_val = float(f.readline().replace('\n', ''))
                max_val = float(f.readline().replace('\n', ''))

            test_data = utils.normalize_arr_with_range(test_data, min_val, max_val)

        if p_mode == 'svdn':
            test_data = utils.normalize_arr(test_data)

    else:

        # check mode to normalize data
        if p_mode == 'svdne':

            # set min_max_filename if custom use
            min_max_file_path = os.path.join(path, p_feature + min_max_ext)

            # need to read min_max_file
            with open(min_max_file_path, 'r') as f:
                min_val = float(f.readline().replace('\n', ''))
                max_val = float(f.readline().replace('\n', ''))

            l_values = utils.normalize_arr_with_range(data, min_val, max_val)

        elif p_mode == 'svdn':
            l_values = utils.normalize_arr(data)
        else:
            l_values = data

        test_data = data[indices]


    # get prediction of model
    if kind_model == 'sklearn':
        prediction = model.predict([test_data])[0]

    if kind_model == 'keras':
        test_data = np.asarray(test_data).reshape(1, len(test_data), 1)
        prediction = model.predict_classes([test_data])[0][0]

    # output expected from others scripts
    print(prediction)

if __name__== "__main__":
    main()
