#! bin/bash

# erase "results/optimization_comparisons.csv" file and write new header
file_path='results/optimization_comparisons.csv'
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
    echo 'data_file; ils_iteration; ls_iteration; best_solution; nb_filters; fitness (roc test);' >> ${file_path}

fi

size=26
feature="filters_statistics"

bash data_processing/generateAndTrain_maxwell_custom_optimization.sh ${size} ${feature} ${data}
