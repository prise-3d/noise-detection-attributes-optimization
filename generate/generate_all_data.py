# main imports
import sys, os, argparse
import numpy as np
import random
import time
import json

# image processing imports
from PIL import Image

from ipfml.processing import transform, segmentation
from ipfml import utils

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt
from data_attributes import get_image_features


# getting configuration information
zone_folder             = cfg.zone_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes values
scenes_list             = cfg.scenes_names
scenes_indexes          = cfg.scenes_indices
choices                 = cfg.normalization_choices
path                    = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_data_folder

generic_output_file_svd = '_random.csv'

def generate_data_svd(data_type, mode):
    """
    @brief Method which generates all .csv files from scenes
    @param data_type,  feature choice
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
    for folder_scene in scenes:

        print(folder_scene)
        scene_path = os.path.join(path, folder_scene)

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

        # get all images of folder
        scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
        number_scene_image = len(scene_images)
            
        for id_img, img_path in enumerate(scene_images):
            
            current_image_postfix = dt.get_scene_image_postfix(img_path)

            current_img = Image.open(img_path)
            img_blocks = segmentation.divide_in_blocks(current_img, (200, 200))

            for id_block, block in enumerate(img_blocks):

                ###########################
                # feature computation part #
                ###########################

                data = get_image_features(data_type, block)

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
                current_file.write(current_image_postfix + ';')

                for val in data:
                    current_file.write(str(val) + ";")

                current_file.write('\n')

            print(data_type + "_" + mode + "_" + folder_scene + " - " + "{0:.2f}".format((id_img + 1) / number_scene_image * 100.) + "%")
            sys.stdout.write("\033[F")

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

    parser = argparse.ArgumentParser(description="Compute and prepare data of feature of all scenes (keep in memory min and max value found)")

    parser.add_argument('--feature', type=str, 
                                    help="feature choice in order to compute data (use 'all' if all features are needed)", 
                                    choices=features_choices)

    args = parser.parse_args()

    p_feature = args.feature

    # generate all or specific feature data
    if p_feature == 'all':
        for m in features_choices:
            generate_data_svd(m, 'svd')
            generate_data_svd(m, 'svdn')
            generate_data_svd(m, 'svdne')
    else:
        generate_data_svd(p_feature, 'svd')
        generate_data_svd(p_feature, 'svdn')
        generate_data_svd(p_feature, 'svdne')

if __name__== "__main__":
    main()
