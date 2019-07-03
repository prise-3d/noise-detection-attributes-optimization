# Noise detection using 26 attributes

## Description

Noise detection on synthesis images with 26 attributes obtained using few filters. 

Filters list:
- average
- wiener
- median
- gaussian
- wavelet

## Requirements

```
pip install -r requirements.txt
```

Generate all needed data for each features (which requires the the whole dataset. In order to get it, you need to contact us).

```bash
python generate/generate_all_data.py --feature all
```

## How to use

### Multiple directories and scripts are available:


- **dataset/\***: all scene files information (zones of each scene, SVD descriptor files information and so on...).
- **train_model.py**: script which is used to run specific model available.
- **data/\***: folder which will contain all *.train* & *.test* files in order to train model.
- **saved_models/*.joblib**: all scikit learn models saved.
- **models_info/***: all markdown files generated to get quick information about model performance and prediction. This folder contains also **model_comparisons.csv** obtained after running runAll_maxwell.sh script.
- **modules/\***: contains all modules usefull for the whole project (such as configuration variables)


**Remark**: Note here that all python script have *--help* command.

```
python generate_data_model.py --help

python generate_data_model.py --output xxxx --interval 0,20  --kind svdne --scenes "A, B, D" --zones "0, 1, 2" --percent 0.7 --sep: --rowindex 1 --custom custom_min_max_filename
```

Parameters explained:
- **output**: filename of data (which will be split into two parts, *.train* and *.test* relative to your choices).
- **interval**: the interval of data you want to use from SVD vector.
- **kind**: kind of data ['svd', 'svdn', 'svdne']; not normalize, normalize vector only and normalize together.
- **scenes**: scenes choice for training dataset.
- **zones**: zones to take for training dataset.
- **percent**: percent of data amount of zone to take (choose randomly) of zone
- **custom**: specify if you want your data normalized using interval and not the whole singular values vector. If it is, the value of this parameter is the output filename which will store the min and max value found. This file will be usefull later to make prediction with model (optional parameter).

### Train model

This is an example of how to train a model

```bash
python train_model.py --data 'data/xxxx' --output 'model_file_to_save' --choice 'model_choice'
```

Expected values for the **choice** parameter are ['svm_model', 'ensemble_model', 'ensemble_model_v2'].

### Predict image using model

Now we have a model trained, we can use it with an image as input:

```bash
python prediction/predict_noisy_image_svd.py --image path/to/image.png --interval "x,x" --model saved_models/xxxxxx.joblib --feature 'lab' --mode 'svdn' --custom 'min_max_filename'
```

- **feature**: feature choice need to be one of the listed above.
- **custom**: specify filename with custom min and max from your data interval. This file was generated using **custom** parameter of one of the **generate_data_model\*.py** script (optional parameter).

The model will return only 0 or 1:
- 1 means noisy image is detected.
- 0 means image seem to be not noisy.

All SVD features developed need:
- Name added into *feature_choices_labels* global array variable of **modules/utils/config.py** file.
- A specification of how you compute the feature into *get_svd_data* method of **modules/utils/data_type.py** file.

### Predict scene using model

Now we have a model trained, we can use it with an image as input:

```bash
python prediction_scene.py --data path/to/xxxx.csv --model saved_model/xxxx.joblib --output xxxxx --scene xxxx
```
**Remark**: *scene* parameter expected need to be the correct name of the Scene.

### Visualize data

All scripts with names **display/display_\*.py** are used to display data information or results.

Just use --help option to get more information.

### Simulate model on scene

All scripts named **prediction/predict_seuil_expe\*.py** are used to simulate model prediction during rendering process. Do not forget the **custom** parameter filename if necessary.

Once you have simulation done. Checkout your **threshold_map/%MODEL_NAME%/simulation\_curves\_zones\_\*/** folder and use it with help of **display_simulation_curves.py** script.

### Others...

All others bash scripts are used to combine and run multiple model combinations...

## License

[The MIT license](https://github.com/prise-3d/Thesis-NoiseDetection-26-attributes/blob/master/LICENSE)
