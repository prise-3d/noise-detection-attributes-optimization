# main imports
import numpy as np
import pandas as pd

import os, sys, argparse

# image processing imports
import matplotlib.pyplot as plt

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from data_attributes import get_image_features

# other variables
learned_zones_folder = cfg.learned_zones_folder
models_name          = cfg.models_names_list

def reconstruct_image(folder_path, model_name, p_limit):
    """
    @brief Method used to display simulation given .csv files
    @param folder_path, folder which contains all .csv files obtained during simulation
    @param model_name, current name of model
    @return nothing
    """

    for name in models_name:
        if name in model_name:
            data_filename = model_name
            learned_zones_folder_path = os.path.join(learned_zones_folder, data_filename)

    data_files = [x for x in os.listdir(folder_path) if '.png' not in x]

    scene_names = [f.split('_')[3] for f in data_files]

    for id, f in enumerate(data_files):

        print(scene_names[id])
        path_file = os.path.join(folder_path, f)

        # TODO : check if necessary to keep information about zone learned when displaying data
        scenes_zones_used_file_path = os.path.join(learned_zones_folder_path, scene_names[id] + '.csv')

        # TODO : find estimated threshold for each zone scene using `data_files` and p_limit
        # TODO : find images for each zone which are attached to this estimated threshold by the model
        # TODO : reconstructed the image using these zones
        # TODO : Save the image with generated name based on scene, model and `p_limit`


def main():

    parser = argparse.ArgumentParser(description="Display simulations curves from simulation data")

    parser.add_argument('--folder', type=str, help='Folder which contains simulations data for scenes')
    parser.add_argument('--model', type=str, help='Name of the model used for simulations')
    parser.add_argument('--limit', type=int, help='Detection limit to target to stop rendering (number of times model tells image has not more noise)')

    args = parser.parse_args()

    p_folder = args.folder
    p_limit  = args.limit
    p_output = args.output

    if args.model:
        p_model = args.model
    else:
        # find p_model from folder if model arg not given (folder path need to have model name)
        if p_folder.split('/')[-1]:
            p_model = p_folder.split('/')[-1]
        else:
            p_model = p_folder.split('/')[-2]
    
    print(p_model)

    reconstruct_image(p_folder, p_model, p_limit)

    print(p_folder)

if __name__== "__main__":
    main()
