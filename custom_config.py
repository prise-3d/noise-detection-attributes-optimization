from modules.config.attributes_config import *

# store all variables from global config
context_vars = vars()

# folders
logs_folder                             = 'logs'
backup_folder                           = 'backups'

## min_max_custom_folder           = 'custom_norm'
## correlation_indices_folder      = 'corr_indices'

# variables
features_choices_labels                 = ['filters_statistics']
optimization_filters_result_filename    = 'optimization_comparisons_filters.csv'
optimization_attributes_result_filename = 'optimization_comparisons_attributes.csv'

filter_reduction_choices                = ['attributes', 'filters']
models_names_list                       = ["svm_model","ensemble_model","ensemble_model_v2","deep_keras"]

## models_names_list               = ["svm_model","ensemble_model","ensemble_model_v2","deep_keras"]
## normalization_choices           = ['svd', 'svdn', 'svdne']

# parameters
## keras_epochs                    = 500
## keras_batch                     = 32
## val_dataset_size                = 0.2