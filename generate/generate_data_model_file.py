# main imports
import sys, os, argparse
import numpy as np
import pandas as pd
import random

# image processing imports
from PIL import Image

from ipfml import utils

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt
from data_attributes import get_image_features


# getting configuration information
learned_folder          = cfg.output_zones_learned
min_max_filename        = cfg.min_max_filename_extension

# define all scenes variables
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

normalization_choices   = cfg.normalization_choices
features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_datasets
custom_min_max_folder   = cfg.min_max_custom_folder
min_max_ext             = cfg.min_max_filename_extension
zones_indices           = cfg.zones_indices

generic_output_file_svd = '_random.csv'

min_value_interval = sys.maxsize
max_value_interval = 0

def construct_new_line(threshold, interval, line, choice, each, norm):
    begin, end = interval

    line_data = line.split(';')
    seuil = line_data[0]
    features = line_data[begin+1:end+1]

    features = [float(m) for id, m in enumerate(features) if id % each == 0 ]

    if norm:
        if choice == 'svdne':
            features = utils.normalize_arr_with_range(features, min_value_interval, max_value_interval)
        if choice == 'svdn':
            features = utils.normalize_arr(features)

    if threshold > int(seuil):
        line = '1'
    else:
        line = '0'

    for val in features:
        line += ';'
        line += str(val)
    line += '\n'

    return line

def get_min_max_value_interval(path, _scenes_list, _interval, _feature):

    global min_value_interval, max_value_interval

    scenes = os.listdir(path)

    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    for folder_scene in scenes:

        # only take care of maxwell scenes
        if folder_scene in _scenes_list:

            scene_path = os.path.join(path, folder_scene)

            zones_folder = []
            # create zones list
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str
                zones_folder.append("zone"+index_str)

            for zone_folder in zones_folder:
                zone_path = os.path.join(scene_path, zone_folder)
                data_filename = _feature + "_svd" + generic_output_file_svd
                data_file_path = os.path.join(zone_path, data_filename)

                # getting number of line and read randomly lines
                f = open(data_file_path)
                lines = f.readlines()

                # check if user select current scene and zone to be part of training data set
                for line in lines:

                    begin, end = _interval

                    line_data = line.split(';')
                    features = line_data[begin+1:end+1]
                    features = [float(m) for m in features]

                    min_value = min(features)
                    max_value = max(features)

                    if min_value < min_value_interval:
                        min_value_interval = min_value

                    if max_value > max_value_interval:
                        max_value_interval = max_value


def generate_data_model(_filename, _data_path, _interval, _choice, _feature, _thresholds, _learned_zones, _step=1, _each=1, _norm=False, _custom=False):

    output_train_filename = os.path.join(output_data_folder, _filename + ".train")
    output_test_filename = os.path.join(output_data_folder,_filename + ".test")

    # create path if not exists
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    # get zone indices
    zones_indices = np.arange(16)

    for folder_scene in _thresholds:

        # get train zones
        train_zones = _learned_zones[folder_scene]
        scene_thresholds = _thresholds[folder_scene]
        scene_path = os.path.join(_data_path, folder_scene)

        for id_zone, index_folder in enumerate(zones_indices):

            index_str = str(index_folder)
            if len(index_str) < 2:
                index_str = "0" + index_str
            current_zone_folder = "zone" + index_str

            zone_path = os.path.join(scene_path, current_zone_folder)

            # if custom normalization choices then we use svd values not already normalized
            if _custom:
                data_filename = _feature + "_svd" + generic_output_file_svd
            else:
                data_filename = _feature + "_" + _choice + generic_output_file_svd

            data_file_path = os.path.join(zone_path, data_filename)

            # getting number of line and read randomly lines
            f = open(data_file_path)
            lines = f.readlines()

            num_lines = len(lines)

            lines_indexes = np.arange(num_lines)
            random.shuffle(lines_indexes)

            counter = 0
            # check if user select current scene and zone to be part of training data set
            for index in lines_indexes:

                image_index = int(lines[index].split(';')[0])

                if image_index % _step == 0:
                    line = construct_new_line(scene_thresholds[id_zone], _interval, lines[index], _choice, _each, _norm)

                    if id_zone in train_zones:
                        train_file.write(line)
                    else:
                        test_file.write(line)

                counter += 1

            f.close()

    train_file.close()
    test_file.close()


def main():

    # getting all params
    parser = argparse.ArgumentParser(description="Generate data for model using correlation matrix information from data")

    parser.add_argument('--output', type=str, help='output file name desired (.train and .test)', required=True)
    parser.add_argument('--data', type=str, help='folder which contains data of dataset', required=True)
    parser.add_argument('--thresholds', type=str, help='file with scene list information and thresholds', required=True)
    parser.add_argument('--selected_zones', type=str, help='file which contains all selected zones of scene', required=True)  
    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"', required=True)
    parser.add_argument('--kind', type=str, help='Kind of normalization level wished', choices=normalization_choices)
    parser.add_argument('--feature', type=str, help='feature data choice', choices=features_choices, required=True)
    parser.add_argument('--step', type=int, help='Photo step to keep for build datasets', default=1)
    parser.add_argument('--each', type=int, help='Each features to keep from interval', default=1)
    parser.add_argument('--custom', type=str, help='Name of custom min max file if use of renormalization of data', default=False)

    args = parser.parse_args()

    p_filename = args.output
    p_data     = args.data
    p_thresholds = args.thresholds
    p_selected_zones = args.selected_zones
    p_interval = list(map(int, args.interval.split(',')))
    p_kind     = args.kind
    p_feature  = args.feature
    p_step     = args.step
    p_each     = args.each
    p_custom   = args.custom

    # 1. retrieve human_thresholds
    human_thresholds = {}

    # extract thresholds
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            # TODO : check if really necessary
            if current_scene != '50_shades_of_grey':
                human_thresholds[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

    # 2. get selected zones
    selected_zones = {}
    with(open(p_selected_zones, 'r')) as f:

        for line in f.readlines():

            data = line.split(';')
            del data[-1]
            scene_name = data[0]
            thresholds = data[1:]

            selected_zones[scene_name] = [ int(t) for t in thresholds ]

    # find min max value if necessary to renormalize data
    if p_custom:
        get_min_max_value_interval(p_data, selected_zones, p_interval, p_feature)

        # write new file to save
        if not os.path.exists(custom_min_max_folder):
            os.makedirs(custom_min_max_folder)

        min_max_folder_path = os.path.join(os.path.dirname(__file__), custom_min_max_folder)
        min_max_filename_path = os.path.join(min_max_folder_path, p_custom)

        with open(min_max_filename_path, 'w') as f:
            f.write(str(min_value_interval) + '\n')
            f.write(str(max_value_interval) + '\n')

    # create database using img folder (generate first time only)
    generate_data_model(p_filename, p_data, p_interval, p_kind, p_feature, human_thresholds, selected_zones, p_step, p_each, p_custom)

if __name__== "__main__":
    main()