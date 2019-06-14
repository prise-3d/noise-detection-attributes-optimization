#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 21:02:42 2018

@author: jbuisine
"""

from __future__ import print_function
import sys, os, argparse
import numpy as np
import random
import time
import json

from modules.utils.data import get_svd_data
from PIL import Image
from ipfml import processing, metrics, utils
from skimage import color

from modules.utils import config as cfg

# getting configuration information
config_filename         = cfg.config_filename
zone_folder             = cfg.zone_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes values
scenes_list             = cfg.scenes_names
scenes_indexes          = cfg.scenes_indices
choices                 = cfg.normalization_choices
path                    = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

metric_choices          = cfg.metric_choices_labels
output_data_folder      = cfg.output_data_folder

generic_output_file_svd = '_random.csv'

def generate_data_svd(data_type, mode):
    """
    @brief Method which generates all .csv files from scenes
    @param data_type,  metric choice
    @param mode, normalization choice
    @return nothing
    """

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    # keep in memory min and max data found from data_type
    min_val_found = sys.maxsize
    max_val_found = 0

    data_min_max_filename = os.path.join(path, data_type + min_max_filename)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        print(folder_scene)
        scene_path = os.path.join(path, folder_scene)

        config_file_path = os.path.join(scene_path, config_filename)

        with open(config_file_path, "r") as config_file:
            last_image_name = config_file.readline().strip()
            prefix_image_name = config_file.readline().strip()
            start_index_image = config_file.readline().strip()
            end_index_image = config_file.readline().strip()
            step_counter = int(config_file.readline().strip())

        # getting output filename
        output_svd_filename = data_type + "_" + mode + generic_output_file_svd

        # construct each zones folder name
        zones_folder = []
        svd_output_files = []

        # get zones list info
        for index in zones:
            index_str = str(index)
            if len(index_str) < 2:
                index_str = "0" + index_str

            current_zone = "zone"+index_str
            zones_folder.append(current_zone)

            zone_path = os.path.join(scene_path, current_zone)
            svd_file_path = os.path.join(zone_path, output_svd_filename)

            # add writer into list
            svd_output_files.append(open(svd_file_path, 'w'))


        current_counter_index = int(start_index_image)
        end_counter_index = int(end_index_image)


        while(current_counter_index <= end_counter_index):

            current_counter_index_str = str(current_counter_index)

            while len(start_index_image) > len(current_counter_index_str):
                current_counter_index_str = "0" + current_counter_index_str

            img_path = os.path.join(scene_path, prefix_image_name + current_counter_index_str + ".png")

            current_img = Image.open(img_path)
            img_blocks = processing.divide_in_blocks(current_img, (200, 200))

            for id_block, block in enumerate(img_blocks):

                ###########################
                # Metric computation part #
                ###########################

                data = get_svd_data(data_type, block)

                ##################
                # Data mode part #
                ##################

                # modify data depending mode
                if mode == 'svdne':

                    # getting max and min information from min_max_filename
                    with open(data_min_max_filename, 'r') as f:
                        min_val = float(f.readline())
                        max_val = float(f.readline())

                    data = utils.normalize_arr_with_range(data, min_val, max_val)

                if mode == 'svdn':
                    data = utils.normalize_arr(data)

                # save min and max found from dataset in order to normalize data using whole data known
                if mode == 'svd':

                    current_min = data.min()
                    current_max = data.max()

                    if current_min < min_val_found:
                        min_val_found = current_min

                    if current_max > max_val_found:
                        max_val_found = current_max

                # now write data into current writer
                current_file = svd_output_files[id_block]

                # add of index
                current_file.write(current_counter_index_str + ';')

                for val in data:
                    current_file.write(str(val) + ";")

                current_file.write('\n')

            start_index_image_int = int(start_index_image)
            print(data_type + "_" + mode + "_" + folder_scene + " - " + "{0:.2f}".format((current_counter_index - start_index_image_int) / (end_counter_index - start_index_image_int)* 100.) + "%")
            sys.stdout.write("\033[F")

            current_counter_index += step_counter

        for f in svd_output_files:
            f.close()

        print('\n')

    # save current information about min file found
    if mode == 'svd':
        with open(data_min_max_filename, 'w') as f:
            f.write(str(min_val_found) + '\n')
            f.write(str(max_val_found) + '\n')

    print("%s_%s : end of data generation\n" % (data_type, mode))


def main():

    parser = argparse.ArgumentParser(description="Compute and prepare data of metric of all scenes (keep in memory min and max value found)")

    parser.add_argument('--metric', type=str, 
                                    help="metric choice in order to compute data (use 'all' if all metrics are needed)", 
                                    choices=metric_choices)

    args = parser.parse_args()

    p_metric = args.metric

    # generate all or specific metric data
    if p_metric == 'all':
        for m in metric_choices:
            generate_data_svd(m, 'svd')
            generate_data_svd(m, 'svdn')
            generate_data_svd(m, 'svdne')
    else:
        generate_data_svd(p_metric, 'svd')
        generate_data_svd(p_metric, 'svdn')
        generate_data_svd(p_metric, 'svdne')

if __name__== "__main__":
    main()
