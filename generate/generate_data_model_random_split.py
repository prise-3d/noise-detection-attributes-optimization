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
learned_folder          = cfg.learned_zones_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes variables
all_scenes_list         = cfg.scenes_names
all_scenes_indices      = cfg.scenes_indices

normalization_choices   = cfg.normalization_choices
path                    = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

renderer_choices        = cfg.renderer_choices
features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_data_folder
custom_min_max_folder   = cfg.min_max_custom_folder
min_max_ext             = cfg.min_max_filename_extension

generic_output_file_svd = '_random.csv'

min_value_interval      = sys.maxsize
max_value_interval      = 0
abs_gap_data            = 100


def construct_new_line(seuil_learned, interval, line, choice, each, norm):
    begin, end = interval

    line_data = line.split(';')
    seuil = line_data[0]
    features = line_data[begin+1:end+1]

    # keep only if modulo result is 0 (keep only each wanted values)
    features = [float(m) for id, m in enumerate(features) if id % each == 0]

    # TODO : check if it's always necessary to do that (loss of information for svd)
    if norm:

        if choice == 'svdne':
            features = utils.normalize_arr_with_range(features, min_value_interval, max_value_interval)
        if choice == 'svdn':
            features = utils.normalize_arr(features)

    if seuil_learned > int(seuil):
        line = '1'
    else:
        line = '0'

    for val in features:
        line += ';'
        line += str(val)
    line += '\n'

    return line

def get_min_max_value_interval(_scenes_list, _interval, _feature):

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

                # if custom normalization choices then we use svd values not already normalized
                data_filename = _feature + "_svd"+ generic_output_file_svd

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


def generate_data_model(_scenes_list, _filename, _interval, _choice, _feature, _scenes, _nb_zones = 4, _percent = 1, _random=0, _step=1, _each=1, _custom = False):

    output_train_filename = _filename + ".train"
    output_test_filename = _filename + ".test"

    if not '/' in output_train_filename:
        raise Exception("Please select filename with directory path to save data. Example : data/dataset")

    # create path if not exists
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    train_file_data = []
    test_file_data  = []

    for folder_scene in _scenes_list:

        scene_path = os.path.join(path, folder_scene)

        zones_indices = zones

        # shuffle list of zones (=> randomly choose zones)
        # only in random mode
        if _random:
            random.shuffle(zones_indices)

        # store zones learned
        learned_zones_indices = zones_indices[:_nb_zones]

        # write into file
        folder_learned_path = os.path.join(learned_folder, _filename.split('/')[1])

        if not os.path.exists(folder_learned_path):
            os.makedirs(folder_learned_path)

        file_learned_path = os.path.join(folder_learned_path, folder_scene + '.csv')

        with open(file_learned_path, 'w') as f:
            for i in learned_zones_indices:
                f.write(str(i) + ';')

        for id_zone, index_folder in enumerate(zones_indices):

            index_str = str(index_folder)
            if len(index_str) < 2:
                index_str = "0" + index_str
            current_zone_folder = "zone" + index_str

            zone_path = os.path.join(scene_path, current_zone_folder)

            # if custom normalization choices then we use svd values not already normalized
            if _custom:
                data_filename = _feature + "_svd"+ generic_output_file_svd
            else:
                data_filename = _feature + "_" + _choice + generic_output_file_svd

            data_file_path = os.path.join(zone_path, data_filename)

            # getting number of line and read randomly lines
            f = open(data_file_path)
            lines = f.readlines()

            num_lines = len(lines)

            # randomly shuffle image
            if _random:
                random.shuffle(lines)

            path_seuil = os.path.join(zone_path, seuil_expe_filename)

            with open(path_seuil, "r") as seuil_file:
                seuil_learned = int(seuil_file.readline().strip())

            counter = 0
            # check if user select current scene and zone to be part of training data set
            for data in lines:

                percent = counter / num_lines
                image_index = int(data.split(';')[0])

                if image_index % _step == 0:

                    with open(path_seuil, "r") as seuil_file:
                        seuil_learned = int(seuil_file.readline().strip())

                    gap_threshold = abs(seuil_learned - image_index)

                    if gap_threshold > abs_gap_data:

                        line = construct_new_line(seuil_learned, _interval, data, _choice, _each, _custom)

                        if id_zone < _nb_zones and folder_scene in _scenes and percent <= _percent:
                            train_file_data.append(line)
                        else:
                            test_file_data.append(line)

                counter += 1

            f.close()

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    for line in train_file_data:
        train_file.write(line)

    for line in test_file_data:
        test_file.write(line)

    train_file.close()
    test_file.close()


def main():

    # getting all params
    parser = argparse.ArgumentParser(description="Generate data for model using correlation matrix information from data")

    parser.add_argument('--output', type=str, help='output file name desired (.train and .test)')
    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--kind', type=str, help='Kind of normalization level wished', choices=normalization_choices)
    parser.add_argument('--feature', type=str, help='feature data choice', choices=features_choices)
    parser.add_argument('--scenes', type=str, help='List of scenes to use for training data')
    parser.add_argument('--nb_zones', type=int, help='Number of zones to use for training data set')
    parser.add_argument('--random', type=int, help='Data will be randomly filled or not', choices=[0, 1])
    parser.add_argument('--percent', type=float, help='Percent of data use for train and test dataset (by default 1)')
    parser.add_argument('--step', type=int, help='Photo step to keep for build datasets', default=1)
    parser.add_argument('--each', type=int, help='Each features to keep from interval', default=1)
    parser.add_argument('--renderer', type=str, help='Renderer choice in order to limit scenes used', choices=renderer_choices, default='all')
    parser.add_argument('--custom', type=str, help='Name of custom min max file if use of renormalization of data', default=False)

    args = parser.parse_args()

    p_filename = args.output
    p_interval = list(map(int, args.interval.split(',')))
    p_kind     = args.kind
    p_feature  = args.feature
    p_scenes   = args.scenes.split(',')
    p_nb_zones = args.nb_zones
    p_random   = args.random
    p_percent  = args.percent
    p_step     = args.step
    p_each     = args.each
    p_renderer = args.renderer
    p_custom   = args.custom


    # list all possibles choices of renderer
    scenes_list = dt.get_renderer_scenes_names(p_renderer)
    scenes_indices = dt.get_renderer_scenes_indices(p_renderer)

    # getting scenes from indexes user selection
    scenes_selected = []

    for scene_id in p_scenes:
        index = scenes_indices.index(scene_id.strip())
        scenes_selected.append(scenes_list[index])

    # find min max value if necessary to renormalize data
    if p_custom:
        get_min_max_value_interval(scenes_list, p_interval, p_feature)

        # write new file to save
        if not os.path.exists(custom_min_max_folder):
            os.makedirs(custom_min_max_folder)

        min_max_filename_path = os.path.join(custom_min_max_folder, p_custom)

        with open(min_max_filename_path, 'w') as f:
            f.write(str(min_value_interval) + '\n')
            f.write(str(max_value_interval) + '\n')

    # create database using img folder (generate first time only)
    generate_data_model(scenes_list, p_filename, p_interval, p_kind, p_feature, scenes_selected, p_nb_zones, p_percent, p_random, p_step, p_each, p_custom)

if __name__== "__main__":
    main()
