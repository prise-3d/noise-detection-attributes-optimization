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

## Project structure

### Link to your dataset

You need database which respects this structure:

- dataset/
  - Scene1/
    - Scene1_00050.png
    - Scene1_00070.png
    - ...
    - Scene1_01180.png
    - Scene1_01200.png
  - Scene2/
    - ...
  - ...

### Code architecture description

- **modules/\***: contains all modules usefull for the whole project (such as configuration variables)
- **analysis/\***: contains all jupyter notebook used for analysis during thesis
- **generate/\***: contains python scripts for generate data from scenes (described later)
- **data_processing/\***: all python scripts for generate custom dataset for models
- **prediction/\***: all python scripts for predict new threshold from computed models
- **data_attributes.py**: files which contains all extracted features implementation from an image.
- **custom_config.py**: override the main configuration project of `modules/config/global_config.py`
- **train_model.py**: script which is used to run specific model available.

### Generated data directories:

- **data/\***: folder which will contain all generated *.train* & *.test* files in order to train model.
- **data/saved_models/\***: all scikit learn or keras models saved.
- **data/models_info/\***: all markdown files generated to get quick information about model performance and prediction obtained after running `run/runAll_*.sh` script.
- **data/results/**:  This folder contains `model_comparisons.csv` file used for store models performance.

## License

[The MIT license](LICENSE)
