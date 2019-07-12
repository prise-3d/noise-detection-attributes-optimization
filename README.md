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

## Project structure

### Link to your dataset

You have to create a symbolic link to your own database which respects this structure:

- dataset/
  - Scene1/
    - zone00/
    - ...
    - zone15/
      - seuilExpe (file which contains threshold samples of zone image perceived by human)
    - Scene1_00050.png
    - Scene1_00070.png
    - ...
    - Scene1_01180.png
    - Scene1_01200.png
  - Scene2/
    - ...
  - ...

Create your symbolic link:

```
ln -s /path/to/your/data dataset
```

### Code architecture description

- **modules/\***: contains all modules usefull for the whole project (such as configuration variables)
- **analysis/\***: contains all jupyter notebook used for analysis during thesis
- **generate/\***: contains python scripts for generate data from scenes (described later)
- **data_processing/\***: all python scripts for generate custom dataset for models
- **prediction/\***: all python scripts for predict new threshold from computed models
- **simulation/\***: contains all bash scripts used for run simulation from models
- **display/\***: contains all python scripts used for display Scene information (such as Singular values...)
- **run/\***: bash scripts to run few step at once : 
  - generate custom dataset
  - train model
  - keep model performance
  - run simulation (if necessary)
- **others/\***: folders which contains others scripts such as script for getting performance of model on specific scene and write it into Mardown file.
- **data_attributes.py**: files which contains all extracted features implementation from an image.
- **custom_config.py**: override the main configuration project of `modules/config/global_config.py`
- **train_model.py**: script which is used to run specific model available.

### Generated data directories:

- **data/\***: folder which will contain all generated *.train* & *.test* files in order to train model.
- **saved_models/\***: all scikit learn or keras models saved.
- **models_info/\***: all markdown files generated to get quick information about model performance and prediction.obtained after running runAll_maxwell.sh script.
- **results/**:  This folder contains `model_comparisons.csv` file used for store models performance.


## How to use ?

**Remark**: Note here that all python script have *--help* command.

```
python generate_data_model.py --help
```

Parameters explained:
- **feature**: feature choice wished
- **output**: filename of data (which will be split into two parts, *.train* and *.test* relative to your choices). Need to be into `data` folder.
- **interval**: the interval of data you want to use from SVD vector.
- **kind**: kind of data ['svd', 'svdn', 'svdne']; not normalize, normalize vector only and normalize together.
- **scenes**: scenes choice for training dataset.
- **zones**: zones to take for training dataset.
- **step**: specify if all pictures are used or not using step process.
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
- Name added into *feature_choices_labels* global array variable of `custom_config.py` file.
- A specification of how you compute the feature into *get_image_features* method of `data_attributes.py` file.

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

## License

[The MIT license](https://github.com/prise-3d/Thesis-NoiseDetection-26-attributes/blob/master/LICENSE)
