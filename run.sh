python generate/generate_all_data.py --feature filters_statistics
python generate/generate_data_model.py --interval 0,26 --kind svdn --feature filters_statistics --scenes A,B,C,D,E,F --zones 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --output data/26_attributes_data --each 1 --kind svdn
python train_model.py --data data/26_attributes_data --output 26_attributes_model --choice svm_model
