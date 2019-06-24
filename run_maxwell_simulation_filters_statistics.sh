#! bin/bash

# file which contains model names we want to use for simulation
simulate_models="simulate_models.csv"

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

size="26"

# for metric in {"lab","mscn","low_bits_2","low_bits_3","low_bits_4","low_bits_5","low_bits_6","low_bits_4_shifted_2","ica_diff","svd_trunc_diff","ipca_diff","svd_reconstruct"}; do
metric="filters_statistics"

for nb_zones in {4,6,8,10,12}; do
    for mode in {"svd","svdn","svdne"}; do
        for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

            FILENAME="data/${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${metric}_${mode}"
            MODEL_NAME="${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${metric}_${mode}"
            CUSTOM_MIN_MAX_FILENAME="N${size}_B0_E${size}_nb_zones_${nb_zones}_${metric}_${mode}_min_max"

            echo $MODEL_NAME

            # only compute if necessary (perhaps server will fall.. Just in case)
            if grep -q "${MODEL_NAME}" "${simulate_models}"; then

                echo "${MODEL_NAME} results already generated..."
            else
                # Use of already generated model
                # python generate_data_model_random.py --output ${FILENAME} --interval "0,${size}" --kind ${mode} --metric ${metric} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 40 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}
                # python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                python predict_seuil_expe_maxwell_curve.py --interval "0,${size}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --metric ${metric} --custom ${CUSTOM_MIN_MAX_FILENAME}

                python save_model_result_in_md_maxwell.py --interval "0,${size}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --metric ${metric}
            fi
        done
    done
done
