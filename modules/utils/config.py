import numpy as np

zone_folder                     = "zone"
output_data_folder              = 'data'
dataset_path                    = 'fichiersSVD_light'
threshold_map_folder            = 'threshold_map'
models_information_folder       = 'models_info'
saved_models_folder             = 'saved_models'
min_max_custom_folder           = 'custom_norm'
learned_zones_folder            = 'learned_zones'
correlation_indices_folder      = 'corr_indices'

csv_model_comparisons_filename  = "models_comparisons.csv"
seuil_expe_filename             = 'seuilExpe'
min_max_filename_extension      = "_min_max_values"
config_filename                 = "config"

models_names_list               = ["svm_model","ensemble_model","ensemble_model_v2","deep_keras"]

# define all scenes values
renderer_choices                = ['all', 'maxwell', 'igloo', 'cycle']

scenes_names                    = ['Appart1opt02', 'Bureau1', 'Cendrier', 'Cuisine01', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre', 'SdbDroite', 'Selles']
scenes_indices                  = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

maxwell_scenes_names            = ['Appart1opt02', 'Cuisine01', 'SdbCentre', 'SdbDroite']
maxwell_scenes_indices          = ['A', 'D', 'G', 'H']

igloo_scenes_names              = ['Bureau1', 'PNDVuePlongeante']
igloo_scenes_indices            = ['B', 'F']

cycle_scenes_names              = ['EchecBas', 'Selles']
cycle_scenes_indices            = ['E', 'I']

normalization_choices           = ['svd', 'svdn', 'svdne']
zones_indices                   = np.arange(16)

metric_choices_labels           = ['lab', 'mscn', 'low_bits_2', 'low_bits_3', 'low_bits_4', 'low_bits_5', 'low_bits_6','low_bits_4_shifted_2', 'sub_blocks_stats', 'sub_blocks_area', 'sub_blocks_stats_reduced', 'sub_blocks_area_normed', 'mscn_var_4', 'mscn_var_16', 'mscn_var_64', 'mscn_var_16_max', 'mscn_var_64_max', 'ica_diff', 'svd_trunc_diff', 'ipca_diff', 'svd_reconstruct', 'highest_sv_std_filters', 'lowest_sv_std_filters']

keras_epochs                    = 500
keras_batch                     = 32
