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
label_freq           = 6

def display_curves(folder_path, model_name):
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

        scenes_zones_used_file_path = os.path.join(learned_zones_folder_path, scene_names[id] + '.csv')

        zones_used = []

        with open(scenes_zones_used_file_path, 'r') as f:
            zones_used = [int(x) for x in f.readline().split(';') if x != '']

        print(zones_used)

        df = pd.read_csv(path_file, header=None, sep=";")

        fig=plt.figure(figsize=(35, 22))
        fig.suptitle("Detection simulation for " + scene_names[id] + " scene", fontsize=20)

        for index, row in df.iterrows():

            row = np.asarray(row)

            threshold = row[2]
            start_index = row[3]
            step_value = row[4]

            counter_index = 0

            current_value = start_index

            while(current_value < threshold):
                counter_index += 1
                current_value += step_value

            fig.add_subplot(4, 4, (index + 1))
            plt.plot(row[5:])

            if index in zones_used:
                ax = plt.gca()
                ax.set_facecolor((0.9, 0.95, 0.95))

            # draw vertical line from (70,100) to (70, 250)
            plt.plot([counter_index, counter_index], [-2, 2], 'k-', lw=2, color='red')

            if index % 4 == 0:
                plt.ylabel('Not noisy / Noisy', fontsize=20)

            if index >= 12:
                plt.xlabel('Samples per pixel', fontsize=20)

            x_labels = [id * step_value + start_index for id, val in enumerate(row[5:]) if id % label_freq == 0]

            x = [v for v in np.arange(0, len(row[5:])+1) if v % label_freq == 0]

            plt.xticks(x, x_labels, rotation=45)
            plt.ylim(-1, 2)

        plt.savefig(os.path.join(folder_path, scene_names[id] + '_simulation_curve.png'))
        #plt.show()

def main():

    parser = argparse.ArgumentParser(description="Display simulations curves from simulation data")

    parser.add_argument('--folder', type=str, help='Folder which contains simulations data for scenes')
    parser.add_argument('--model', type=str, help='Name of the model used for simulations')

    args = parser.parse_args()

    p_folder = args.folder

    if args.model:
        p_model = args.model
    else:
        # find p_model from folder if model arg not given (folder path need to have model name)
        if p_folder.split('/')[-1]:
            p_model = p_folder.split('/')[-1]
        else:
            p_model = p_folder.split('/')[-2]
    
    print(p_model)

    display_curves(p_folder, p_model)

    print(p_folder)

if __name__== "__main__":
    main()
