# main imports
import numpy as np
import sys, os, argparse
import subprocess
import time

# models imports
from sklearn.externals import joblib

# image processing imports
from PIL import Image

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

# variables and parameters
threshold_map_folder      = cfg.threshold_map_folder
threshold_map_file_prefix = cfg.threshold_map_folder + "_"

markdowns_folder          = cfg.models_information_folder
zones                     = cfg.zones_indices

current_dirpath = os.getcwd()

def main():

    parser = argparse.ArgumentParser(description="Display SVD data of scene zone")

    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--model', type=str, help='.joblib or .json file (sklearn or keras model)')
    parser.add_argument('--feature', type=str, help='Feature data choice', choices=cfg.features_choices_labels)
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=cfg.normalization_choices)

    args = parser.parse_args()
    
    p_interval   = list(map(int, args.interval.split(',')))
    p_model_file = args.model
    p_metric     = args.metric
    p_mode       = args.mode


    # call model and get global result in scenes

    begin, end = p_interval

    bash_cmd = "bash others/testModelByScene.sh '" + str(begin) + "' '" + str(end) + "' '" + p_model_file + "' '" + p_mode + "' '" + p_metric + "'"
    print(bash_cmd)

    ## call command ##
    p = subprocess.Popen(bash_cmd, stdout=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()

    ## Wait for result ##
    p_status = p.wait()

    if not os.path.exists(markdowns_folder):
        os.makedirs(markdowns_folder)

    # get model name to construct model
    md_model_path = os.path.join(markdowns_folder, p_model_file.split('/')[-1].replace('.joblib', '.md'))

    with open(md_model_path, 'w') as f:
        f.write(output.decode("utf-8"))

        # read each threshold_map information if exists
        model_map_info_path = os.path.join(threshold_map_folder, p_model_file.replace('saved_models/', ''))

        if not os.path.exists(model_map_info_path):
            f.write('\n\n No threshold map information')
        else:
            maps_files = os.listdir(model_map_info_path)

            # get all map information
            for t_map_file in maps_files:

                file_path = os.path.join(model_map_info_path, t_map_file)
                with open(file_path, 'r') as map_file:

                    title_scene =  t_map_file.replace(threshold_map_file_prefix, '')
                    f.write('\n\n## ' + title_scene + '\n')
                    content = map_file.readlines()

                    # getting each map line information
                    for line in content:
                        f.write(line)

        f.close()

if __name__== "__main__":
    main()
