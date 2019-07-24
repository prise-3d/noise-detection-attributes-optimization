#! bin/bash

# erase "results/models_comparisons.csv" file and write new header
file_path='results/models_comparisons.csv'
list="all, center, split"

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need argument from [${list}]"
    exit 1
fi

if [[ "$1" =~ ^(all|center|split)$ ]]; then
    echo "$1 is in the list"
else
    echo "$1 is not in the list"
fi

data=$1
erased=$2

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p results
    touch ${file_path}

    # add of header
    echo 'model_name; vector_size; start; end; nb_zones; feature; mode; tran_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; recall_train; roc_auc_train; F1_val; recall_val; roc_auc_val; F1_test; recall_test; roc_auc_test; F1_all; recall_all; roc_auc_all;' >> ${file_path}

fi

size=26
feature="filters_statistics"

bash data_processing/generateAndTrain_maxwell_custom.sh ${size} ${feature} ${data}
