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
from modules.utils import data as dt

# other variables
learned_zones_folder = cfg.learned_zones_folder
models_name          = cfg.models_names_list

# utils information
zone_width, zone_height = (200, 200)
scene_width, scene_height = (800, 800)
nb_x_parts = math.floor(scene_width / zone_width)


def reconstruct_image(scene_name, output):
    """
    @brief Method used to display simulation given .csv files
    @param scene_name, scene name used
    @param output, the output filename
    @return nothing
    """

    # compute zone start index
    zones_coordinates = []
    for zone_index in cfg.zones_indices:
        x_zone = (zone_index % nb_x_parts) * zone_width
        y_zone = (math.floor(zone_index / nb_x_parts)) * zone_height

        zones_coordinates.append((x_zone, y_zone))

    scene_folder = os.path.join(cfg.dataset_path, scene_name)

    folder_scene_elements = os.listdir(scene_folder)

    zones_folder = [zone for zone in folder_scene_elements if 'zone' in zone]
    zones_folder = sorted(zones_folder)

    scenes_images = [img for img in folder_scene_elements if cfg.scene_image_extension in img]
    scenes_images = sorted(scenes_images)

    # 1. find thresholds from scene
    human_thresholds = []

    for zone_folder in zones_folder:
        zone_path = os.path.join(scene_folder, zone_folder)
        
        with open(os.path.join(zone_path, cfg.seuil_expe_filename)) as f:
            human_thresholds.append(int(f.readline()))

    # 2. find images for each zone which are attached to these human thresholds by the model
    zone_images_index = []

    for threshold in human_thresholds:

        current_image_index = 0

        for image_name in scenes_images:

            image_quality = dt.get_scene_image_quality(image_name)

            if image_quality > threshold:
                current_image_index = image_quality
                break


        str_index = str(current_image_index)
        while len(str_index) < 5:
            str_index = "0" + str_index

        zone_images_index.append(str_index)

    images_zones = []
    line_images_zones = []
    # get image using threshold by zone
    for id, zone_index in enumerate(zone_images_index):
        filtered_images = [img for img in scenes_images if zone_index in img]
        
        if len(filtered_images) > 0:
            image_name = filtered_images[0]
        else:
            image_name = scenes_images[-1]
        
        image_path = os.path.join(scene_folder, image_name)
        selected_image = Image.open(image_path)

        x_zone, y_zone = zones_coordinates[id]
        zone_image = np.array(selected_image)[y_zone:y_zone+zone_height, x_zone:x_zone+zone_width]
        line_images_zones.append(zone_image)

        if int(id + 1) % int(scene_width / zone_width) == 0:
            images_zones.append(np.concatenate(line_images_zones, axis=1))
            line_images_zones = []


    # 3. reconstructed the image using these zones
    reconstructed_image = np.concatenate(images_zones, axis=0)

    # 4. Save the image with generated name based on scene
    reconstructed_pil_img = Image.fromarray(reconstructed_image)

    folders = output.split('/')
    if len(folders) > 1:
        output_folder = '/'.join(folders[:len(folders) - 1])
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    reconstructed_pil_img.save(output)


def main():

    parser = argparse.ArgumentParser(description="Compute and save reconstructed images from human thresholds")

    parser.add_argument('--scene', type=str, help='Scene index to use', choices=cfg.scenes_indices)
    parser.add_argument('--output', type=str, help='Output reconstructed image path and filename')

    args = parser.parse_args()

    p_scene = args.scene
    p_output = args.output
    
    scenes_list = cfg.scenes_names
    scenes_indices = cfg.scenes_indices

    scene_index = scenes_indices.index(p_scene.strip())
    scene_name = scenes_list[scene_index]

    reconstruct_image(scene_name, p_output)

if __name__== "__main__":
    main()
