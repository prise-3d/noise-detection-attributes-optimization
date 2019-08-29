#! bin/bash

# file which contains model names we want to use for simulation
simulate_models="simulate_models_all.csv"

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

size="26"

feature="filters_statistics"

for nb_zones in {4,6,8,10,12}; do
    for mode in {"svd","svdn","svdne"}; do
        for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do
            for data in {"all","center","split"}; do

                FILENAME="data/${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}_${data}"
                MODEL_NAME="${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}_${data}"
                CUSTOM_MIN_MAX_FILENAME="N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}_${data}_min_max"

                # only compute if necessary (perhaps server will fall.. Just in case)
                if grep -q "${FILENAME}" "${simulate_models}"; then

                    echo "Found ${FILENAME}"
                    line=$(grep -n ${FILENAME} ${simulate_models})

                    # extract solution
                    IFS=\; read -a fields <<<"$line"

                    SOLUTION=${fields[1]}

                    echo "Run simulation for ${MODEL_NAME}... with ${SOLUTION}"

                    # Use of already generated model
                    python generate/generate_data_model_random_${data}.py --output ${FILENAME} --interval "0,${size}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}
                    python train_model_filters.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model} --solution "${SOLUTION}"

                    python prediction/predict_seuil_expe_maxwell_curve_filters.py --solution "${SOLUTION}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature} --custom ${CUSTOM_MIN_MAX_FILENAME}

                    #python others/save_model_result_in_md_maxwell.py --solution "${SOLUTION}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature}
                fi
            done
        done
    done
done