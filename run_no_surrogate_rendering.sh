#! /bin/bash

# default param
ILS=10
LS=100
SS=1050 # Keep aware of surrogate never started
LENGTH=32 # number of features
POP=20
ORDER=1
TRAIN_EVERY=10


#output="rendering-attributes-ILS_${ILS}-POP_${POP}-LS_${LS}-SS_${SS}-SO_${ORDER}-SE_${TRAIN_EVERY}"
DATASET="rnn/data/datasets/features-selection-rendering-scaled/features-selection-rendering-scaled"

for run in {1,2,3,4,5};
do
    # for POP in {20,60,100};
    # do
        #for ORDER in {1,2};
        #for ORDER in {1};
        #do
            #for LS in {100,500,1000};
            #for LS in {10};
            #do
                output="no-rendering-attributes-POP_${POP}-LS_${LS}-RUN_${run}"
                echo "Run optim attributes using: ${output}"
                python find_best_attributes_no_surrogate.py --data ${DATASET} --start_surrogate ${SS} --length 32 --ils ${ILS} --ls ${LS} --pop ${POP} --order ${ORDER} --train_every ${TRAIN_EVERY}  --output ${output}
            #done
        #done
    # done
done
