# main imports
import sys, os, argparse
import subprocess
import time
import numpy as np

# image processing imports
from ipfml.processing import segmentation
from PIL import Image

# models imports
from sklearn.externals import joblib

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt


# variables and parameters
scenes_path               = cfg.dataset_path
min_max_filename          = cfg.min_max_filename_extension
threshold_expe_filename   = cfg.seuil_expe_filename

threshold_map_folder      = cfg.threshold_map_folder
threshold_map_file_prefix = cfg.threshold_map_folder + "_"

zones                     = cfg.zones_indices
normalization_choices     = cfg.normalization_choices
features_choices          = cfg.features_choices_labels

tmp_filename              = '/tmp/__model__img_to_predict.png'

current_dirpath = os.getcwd()

def main():

    p_custom = False

    parser = argparse.ArgumentParser(description="Script which predicts threshold using specific model")

    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--model', type=str, help='.joblib or .json file (sklearn or keras model)')
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=normalization_choices)
    parser.add_argument('--feature', type=str, help='Feature data choice', choices=features_choices)
    parser.add_argument('--limit_detection', type=int, help='Specify number of same prediction to stop threshold prediction', default=2)
    parser.add_argument('--custom', type=str, help='Name of custom min max file if use of renormalization of data', default=False)

    args = parser.parse_args()

    p_interval   = list(map(int, args.interval.split(',')))
    p_model_file = args.model
    p_mode       = args.mode
    p_feature     = args.feature
    p_limit      = args.limit
    p_custom     = args.custom

    scenes = os.listdir(scenes_path)
    scenes = [s for s in scenes if not min_max_filename in s]

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        print(folder_scene)

        scene_path = os.path.join(scenes_path, folder_scene)

        threshold_expes = []
        threshold_expes_detected = []
        threshold_expes_counter = []
        threshold_expes_found = []

            # get all images of folder
        scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])

        start_quality_image = dt.get_scene_image_quality(scene_images[0])
        end_quality_image   = dt.get_scene_image_quality(scene_images[-1])
       
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
                threshold_expes_detected.append(False)
                threshold_expes_counter.append(0)
                threshold_expes_found.append(end_quality_image) # by default use max

        check_all_done = False

        # for each images
        for img_path in scene_images:

            current_img = Image.open(img_path)
            current_quality_image = dt.get_scene_image_quality(img_path)
            current_image_potfix = dt.get_scene_image_postfix(img_path)

            img_blocks = segmentation.divide_in_blocks(current_img, (200, 200))
            current_img = Image.open(img_path)
            img_blocks = segmentation.divide_in_blocks(current_img, (200, 200))

            check_all_done = all(d == True for d in threshold_expes_detected)

            if check_all_done:
                break

            for id_block, block in enumerate(img_blocks):

                # check only if necessary for this scene (not already detected)
                if not threshold_expes_detected[id_block]:

                    tmp_file_path = tmp_filename.replace('__model__',  p_model_file.split('/')[-1].replace('.joblib', '_'))
                    block.save(tmp_file_path)

                    python_cmd = "python prediction/predict_noisy_image_svd.py --image " + tmp_file_path + \
                                    " --interval '" + p_interval + \
                                    "' --model " + p_model_file  + \
                                    " --mode " + p_mode + \
                                    " --feature " + p_feature

                    # specify use of custom file for min max normalization
                    if p_custom:
                        python_cmd = python_cmd + ' --custom ' + p_custom


                    ## call command ##
                    p = subprocess.Popen(python_cmd, stdout=subprocess.PIPE, shell=True)

                    (output, err) = p.communicate()

                    ## Wait for result ##
                    p_status = p.wait()

                    prediction = int(output)

                    if prediction == 0:
                        threshold_expes_counter[id_block] = threshold_expes_counter[id_block] + 1
                    else:
                        threshold_expes_counter[id_block] = 0

                    if threshold_expes_counter[id_block] == p_limit:
                        threshold_expes_detected[id_block] = True
                        threshold_expes_found[id_block] = current_quality_image

                    print(str(id_block) + " : " + current_image_potfix + "/" + str(threshold_expes[id_block]) + " => " + str(prediction))

            print("------------------------")
            print("Scene " + str(id_scene + 1) + "/" + str(len(scenes)))
            print("------------------------")

        # end of scene => display of results

        # construct path using model name for saving threshold map folder
        model_treshold_path = os.path.join(threshold_map_folder, p_model_file.split('/')[-1].replace('.joblib', ''))

        # create threshold model path if necessary
        if not os.path.exists(model_treshold_path):
            os.makedirs(model_treshold_path)

        abs_dist = []

        map_filename = os.path.join(model_treshold_path, threshold_map_file_prefix + folder_scene)
        f_map = open(map_filename, 'w')

        line_information = ""

        # default header
        f_map.write('|  |    |    |  |\n')
        f_map.write('---|----|----|---\n')
        for id, threshold in enumerate(threshold_expes_found):

            line_information += str(threshold) + " / " + str(threshold_expes[id]) + " | "
            abs_dist.append(abs(threshold - threshold_expes[id]))

            if (id + 1) % 4 == 0:
                f_map.write(line_information + '\n')
                line_information = ""

        f_map.write(line_information + '\n')

        min_abs_dist = min(abs_dist)
        max_abs_dist = max(abs_dist)
        avg_abs_dist = sum(abs_dist) / len(abs_dist)

        f_map.write('\nScene information : ')
        f_map.write('\n- BEGIN : ' + str(start_quality_image))
        f_map.write('\n- END : ' + str(end_quality_image))

        f_map.write('\n\nDistances information : ')
        f_map.write('\n- MIN : ' + str(min_abs_dist))
        f_map.write('\n- MAX : ' + str(max_abs_dist))          
        f_map.write('\n- AVG : ' + str(avg_abs_dist))

        f_map.write('\n\nOther information : ')
        f_map.write('\n- Detection limit : ' + str(p_limit))

        # by default print last line
        f_map.close()

        print("Scene " + str(id_scene + 1) + "/" + str(len(scenes)) + " Done..")
        print("------------------------")

        time.sleep(1)


if __name__== "__main__":
    main()
