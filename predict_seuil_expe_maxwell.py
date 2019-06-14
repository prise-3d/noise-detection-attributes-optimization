from sklearn.externals import joblib

import numpy as np

from ipfml import processing
from PIL import Image

import sys, os, getopt
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

tmp_filename              = '/tmp/__model__img_to_predict.png'

current_dirpath = os.getcwd()

def main():

    # by default..
    p_custom = False

    # TODO : use of argparse

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python predict_seuil_expe_maxwell.py --interval "0,20" --model path/to/xxxx.joblib --mode svdn --metric lab --limit_detection xx --custom min_max_filename')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:m:o:l:c", ["help=", "interval=", "model=", "mode=", "metric=", "limit_detection=", "custom="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python predict_seuil_expe_maxwell.py --interval "xx,xx" --model path/to/xxxx.joblib --mode svdn --metric lab --limit_detection xx --custom min_max_filename')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python predict_seuil_expe_maxwell.py --interval "xx,xx" --model path/to/xxxx.joblib --mode svdn --metric lab --limit_detection xx --custom min_max_filename')
            sys.exit()
        elif o in ("-t", "--interval"):
            p_interval = a
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--mode"):
            p_mode = a

            if p_mode != 'svdn' and p_mode != 'svdne' and p_mode != 'svd':
                assert False, "Mode not recognized"

        elif o in ("-m", "--metric"):
            p_metric = a
        elif o in ("-l", "--limit_detection"):
            p_limit = int(a)
        elif o in ("-c", "--custom"):
            p_custom = a
        else:
            assert False, "unhandled option"

    scenes = os.listdir(scenes_path)

    scenes = [s for s in scenes if s in maxwell_scenes]

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
            threshold_expes_detected = []
            threshold_expes_counter = []
            threshold_expes_found = []

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
                    threshold_expes_found.append(int(end_index_image)) # by default use max

            current_counter_index = int(start_index_image)
            end_counter_index = int(end_index_image)

            print(current_counter_index)
            check_all_done = False

            while(current_counter_index <= end_counter_index and not check_all_done):

                current_counter_index_str = str(current_counter_index)

                while len(start_index_image) > len(current_counter_index_str):
                    current_counter_index_str = "0" + current_counter_index_str

                img_path = os.path.join(scene_path, prefix_image_name + current_counter_index_str + ".png")

                current_img = Image.open(img_path)
                img_blocks = processing.divide_in_blocks(current_img, (200, 200))


                check_all_done = all(d == True for d in threshold_expes_detected)

                for id_block, block in enumerate(img_blocks):

                    # check only if necessary for this scene (not already detected)
                    if not threshold_expes_detected[id_block]:

                        tmp_file_path = tmp_filename.replace('__model__',  p_model_file.split('/')[-1].replace('.joblib', '_'))
                        block.save(tmp_file_path)

                        python_cmd = "python predict_noisy_image_svd.py --image " + tmp_file_path + \
                                        " --interval '" + p_interval + \
                                        "' --model " + p_model_file  + \
                                        " --mode " + p_mode + \
                                        " --metric " + p_metric

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
                            threshold_expes_found[id_block] = current_counter_index

                        print(str(id_block) + " : " + str(current_counter_index) + "/" + str(threshold_expes[id_block]) + " => " + str(prediction))

                current_counter_index += step_counter
                print("------------------------")
                print("Scene " + str(id_scene + 1) + "/" + str(len(maxwell_scenes)))
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
            f_map.write('\n- BEGIN : ' + str(start_index_image))
            f_map.write('\n- END : ' + str(end_index_image))

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

            time.sleep(10)


if __name__== "__main__":
    main()
