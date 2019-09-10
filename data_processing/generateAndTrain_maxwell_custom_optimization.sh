#! bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of vector size"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No argument supplied"
    echo "Need of feature information"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No argument supplied"
    echo "Need of kind of data to use"
    exit 1
fi

if [ -z "$4" ]
  then
    echo "No argument supplied"
    echo "Use of filters or attributes"
    exit 1
fi


size=$1
feature=$2
data=$3
filter=$4


# selection of four scenes (only maxwell)
scenes="A, D, G, H"
result_filename="results/optimization_comparisons_${filter}.csv"
start=0
end=$size

#for nb_zones in {4,6,8,10,12}; do
for nb_zones in {10,12}; do

    for mode in {"svd","svdn","svdne"}; do
        #for model in {"svm_model","ensemble_model",""}; do
        model="svm_model"

            FILENAME="data/${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}_${filter}"
            MODEL_NAME="${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}_${filter}"
            CUSTOM_MIN_MAX_FILENAME="N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}_${data}_${filter}_min_max"

            echo $FILENAME

            # only compute if necessary (perhaps server will fall.. Just in case)
            if grep -q "${MODEL_NAME}" "${result_filename}"; then

                echo "${MODEL_NAME} results already generated..."
            else
                python generate/generate_data_model_random_${data}.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 40 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}
                
                echo "Train ${MODEL_NAME}"
                python find_best_${filter}.py --data ${FILENAME} --choice ${model}
            fi
        #done
    done
done