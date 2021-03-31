#! /bin/bash

# default param
ILS=10000
LS=100
SS=50
LENGTH=32 # number of features
POP=100
ORDER=2
TRAIN_EVERY=50


#output="rendering-attributes-ILS_${ILS}-POP_${POP}-LS_${LS}-SS_${SS}-SO_${ORDER}-SE_${TRAIN_EVERY}"
DATASET="rnn/data/datasets/features-selection-rendering-scaled/features-selection-rendering-scaled"

for run in {1,2,3,4,5,6,7,8,9,10};
    for POP in {20,60,100};
    do
        for ORDER in {1,2,3};
        do
            for LS in {1000,5000,10000};
            do
                output="rendering-attributes-POP_${POP}-LS_${LS}-SS_${SS}-SO_${ORDER}-SE_${TRAIN_EVERY}-RUN_${run}"
                echo "Run optim attributes using: ${output}"
                python find_best_attributes_surrogate.py --data ${DATASET} --start_surrogate ${SS} --length 32 --ils ${ILS} --ls ${LS} --pop ${POP} --order ${ORDER} --train_every ${TRAIN_EVERY}  --output ${output}
            done
        done
    done
done
