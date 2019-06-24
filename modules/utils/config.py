import numpy as np

zone_folder                     = "zone"
output_data_folder              = 'data'
dataset_path                    = 'dataset'
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

metric_choices_labels           = ['filters_statistics']

keras_epochs                    = 100
keras_batch                     = 32
