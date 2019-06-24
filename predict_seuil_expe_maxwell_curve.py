from sklearn.externals import joblib

import numpy as np

from ipfml import processing
from PIL import Image

import sys, os, argparse
import subprocess
import time

from modules.utils import config as cfg

config_filename           = cfg.config_filename
scenes_path               = cfg.dataset_path
min_max_filename          = cfg.min_max_filename_extension
threshold_expe_filename   = cfg.seuil_expe_filename

threshold_map_folder      = cfg.threshold_map_folder
threshold_map_file_prefix = cfg.threshold_map_folder + "_"

zones                     = cfg.zones_indices
maxwell_scenes            = cfg.maxwell_scenes_names
normalization_choices     = cfg.normalization_choices
metric_choices            = cfg.metric_choices_labels

simulation_curves_zones   = "simulation_curves_zones_"
tmp_filename              = '/tmp/__model__img_to_predict.png'

current_dirpath = os.getcwd()


def main():

    p_custom = False
        
    parser = argparse.ArgumentParser(description="Script which predicts threshold using specific model")

    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--model', type=str, help='.joblib or .json file (sklearn or keras model)')
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=normalization_choices)
    parser.add_argument('--metric', type=str, help='Metric data choice', choices=metric_choices)
    #parser.add_argument('--limit_detection', type=int, help='Specify number of same prediction to stop threshold prediction', default=2)
    parser.add_argument('--custom', type=str, help='Name of custom min max file if use of renormalization of data', default=False)

    args = parser.parse_args()

    # keep p_interval as it is
    p_interval   = args.interval
    p_model_file = args.model
    p_mode       = args.mode
    p_metric     = args.metric
    #p_limit      = args.limit
    p_custom     = args.custom

    scenes = os.listdir(scenes_path)
    scenes = [s for s in scenes if s in maxwell_scenes]

    print(scenes)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        # only take in consideration maxwell scenes
        if folder_scene in maxwell_scenes:

            print(folder_scene)

            scene_path = os.path.join(scenes_path, folder_scene)

            config_path = os.path.join(scene_path, config_filename)

            with open(config_path, "r") as config_file:
                last_image_name = config_file.readline().strip()
                prefix_image_name = config_file.readline().strip()
                start_index_image = config_file.readline().strip()
                end_index_image = config_file.readline().strip()
                step_counter = int(config_file.readline().strip())

            threshold_expes = []
            threshold_expes_found = []
            block_predictions_str = []

            # get zones list info
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str
                zone_folder = "zone"+index_str

                threshold_path_file = os.path.join(os.path.join(scene_path, zone_folder), threshold_expe_filename)

                with open(threshold_path_file) as f:
                    threshold = int(f.readline())
                    threshold_expes.append(threshold)

                    # Initialize default data to get detected model threshold found
                    threshold_expes_found.append(int(end_index_image)) # by default use max

                block_predictions_str.append(index_str + ";" + p_model_file + ";" + str(threshold) + ";" + str(start_index_image) + ";" + str(step_counter))

            current_counter_index = int(start_index_image)
            end_counter_index = int(end_index_image)

            print(current_counter_index)

            while(current_counter_index <= end_counter_index):

                current_counter_index_str = str(current_counter_index)

                while len(start_index_image) > len(current_counter_index_str):
                    current_counter_index_str = "0" + current_counter_index_str

                img_path = os.path.join(scene_path, prefix_image_name + current_counter_index_str + ".png")

                current_img = Image.open(img_path)
                img_blocks = processing.divide_in_blocks(current_img, (200, 200))

                for id_block, block in enumerate(img_blocks):

                    # check only if necessary for this scene (not already detected)
                    #if not threshold_expes_detected[id_block]:

                        tmp_file_path = tmp_filename.replace('__model__',  p_model_file.split('/')[-1].replace('.joblib', '_'))
                        block.save(tmp_file_path)

                        python_cmd_line = "python predict_noisy_image_svd.py --image {0} --interval '{1}' --model {2} --mode {3} --metric {4}"
                        python_cmd = python_cmd_line.format(tmp_file_path, p_interval, p_model_file, p_mode, p_metric) 

                        # specify use of custom file for min max normalization
                        if p_custom:
                            python_cmd = python_cmd + ' --custom ' + p_custom

                        ## call command ##
                        p = subprocess.Popen(python_cmd, stdout=subprocess.PIPE, shell=True)

                        (output, err) = p.communicate()

                        ## Wait for result ##
                        p_status = p.wait()

                        prediction = int(output)

                        # save here in specific file of block all the predictions done
                        block_predictions_str[id_block] = block_predictions_str[id_block] + ";" + str(prediction)

                        print(str(id_block) + " : " + str(current_counter_index) + "/" + str(threshold_expes[id_block]) + " => " + str(prediction))

                current_counter_index += step_counter
                print("------------------------")
                print("Scene " + str(id_scene + 1) + "/" + str(len(scenes)))
                print("------------------------")

            # end of scene => display of results

            # construct path using model name for saving threshold map folder
            model_threshold_path = os.path.join(threshold_map_folder, p_model_file.split('/')[-1].replace('.joblib', ''))

            # create threshold model path if necessary
            if not os.path.exists(model_threshold_path):
                os.makedirs(model_threshold_path)

            map_filename = os.path.join(model_threshold_path, simulation_curves_zones + folder_scene)
            f_map = open(map_filename, 'w')

            for line in block_predictions_str:
                f_map.write(line + '\n')
            f_map.close()

            print("Scene " + str(id_scene + 1) + "/" + str(len(maxwell_scenes)) + " Done..")
            print("------------------------")

            print("Model predictions are saved into %s" % map_filename)
            time.sleep(10)


if __name__== "__main__":
    main()