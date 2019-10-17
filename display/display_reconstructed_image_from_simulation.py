# main imports
import numpy as np
import pandas as pd
import math
import time

import os, sys, argparse

# image processing imports
import matplotlib.pyplot as plt
from PIL import Image

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from data_attributes import get_image_features

# other variables
learned_zones_folder = cfg.learned_zones_folder
models_name          = cfg.models_names_list

# utils information
zone_width, zone_height = (200, 200)
scene_width, scene_height = (800, 800)
nb_x_parts = math.floor(scene_width / zone_width)


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

    # compute zone start index
    zones_coordinates = []
    for index, zone_index in enumerate(cfg.zones_indices):
        x_zone = (zone_index % nb_x_parts) * zone_width
        y_zone = (math.floor(zone_index / nb_x_parts)) * zone_height

        zones_coordinates.append((x_zone, y_zone))

    print(zones_coordinates)

    for id, f in enumerate(data_files):

        scene_name = scene_names[id]
        path_file = os.path.join(folder_path, f)

        # TODO : check if necessary to keep information about zone learned when displaying data
        scenes_zones_used_file_path = os.path.join(learned_zones_folder_path, scene_name + '.csv')

        zones_used = []

        if os.path.exists(scenes_zones_used_file_path):
            with open(scenes_zones_used_file_path, 'r') as f:
                zones_used = [int(x) for x in f.readline().split(';') if x != '']

        # 1. find estimated threshold for each zone scene using `data_files` and p_limit
        model_thresholds = []
        df = pd.read_csv(path_file, header=None, sep=";")

        for index, row in df.iterrows():

            row = np.asarray(row)

            threshold = row[2]
            start_index = row[3]
            step_value = row[4]
            rendering_predictions = row[5:]

            nb_generated_image = 0
            nb_not_noisy_prediction = 0

            for prediction in rendering_predictions:
                
                if int(prediction) == 0:
                    nb_not_noisy_prediction += 1
                else:
                    nb_not_noisy_prediction = 0

                # exit loop if limit is targeted
                if nb_not_noisy_prediction >= p_limit:
                    break

                nb_generated_image += 1
            
            current_threshold = start_index + step_value * nb_generated_image
            model_thresholds.append(current_threshold)

        # 2. find images for each zone which are attached to this estimated threshold by the model

        zone_images_index = []

        for est_threshold in model_thresholds:

            str_index = str(est_threshold)
            while len(str_index) < 5:
                str_index = "0" + str_index

            zone_images_index.append(str_index)

        scene_folder = os.path.join(cfg.dataset_path, scene_name)
        
        scenes_images = [img for img in os.listdir(scene_folder) if cfg.scene_image_extension in img]
        scenes_images = sorted(scenes_images)

        images_zones = []
        line_images_zones = []
        # get image using threshold by zone
        for id, zone_index in enumerate(zone_images_index):
            filtered_images = [img for img in scenes_images if zone_index in img]
            
            if len(filtered_images) > 0:
                image_name = filtered_images[0]
            else:
                image_name = scenes_images[-1]
            
            #print(image_name)
            image_path = os.path.join(scene_folder, image_name)
            selected_image = Image.open(image_path)

            x_zone, y_zone = zones_coordinates[id]
            zone_image = np.array(selected_image)[y_zone:y_zone+zone_height, x_zone:x_zone+zone_width]
            line_images_zones.append(zone_image)

            if int(id + 1) % int(scene_width / zone_width) == 0:
                images_zones.append(np.concatenate(line_images_zones, axis=1))
                print(len(line_images_zones))
                line_images_zones = []


        # 3. reconstructed the image using these zones
        reconstructed_image = np.concatenate(images_zones, axis=0)

        # 4. Save the image with generated name based on scene, model and `p_limit`
        reconstructed_pil_img = Image.fromarray(reconstructed_image)

        output_path = os.path.join(folder_path, scene_names[id] + '_reconstruction_limit_' + str(p_limit) + '.png')

        reconstructed_pil_img.save(output_path)


def main():

    parser = argparse.ArgumentParser(description="Display simulations curves from simulation data")

    parser.add_argument('--folder', type=str, help='Folder which contains simulations data for scenes')
    parser.add_argument('--model', type=str, help='Name of the model used for simulations')
    parser.add_argument('--limit', type=int, help='Detection limit to target to stop rendering (number of times model tells image has not more noise)')

    args = parser.parse_args()

    p_folder = args.folder
    p_limit  = args.limit

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
