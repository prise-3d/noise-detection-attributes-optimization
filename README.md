# Noise detection using SVM

## Requirements

```
pip install -r requirements.txt
```

Generate all needed data for each metrics (which requires the the whole dataset. In order to get it, you need to contact us).

```bash
python generate_all_data.py --metric all
```

For noise detection, many metrics are available:
- lab
- mscn
- mscn_revisited
- low_bits_2
- low_bits_4
- low_bits_5
- low_bits_6
- low_bits_4_shifted_2

You can also specify metric you want to compute and image step to avoid some images:
```bash
python generate_all_data.py --metric mscn --step 50
```

- **step**: keep only image if image id % 50 == 0 (assumption is that keeping spaced data will let model better fit).

## How to use

### Multiple directories and scripts are available:


- **fichiersSVD_light/\***: all scene files information (zones of each scene, SVD descriptor files information and so on...).
- **train_model.py**: script which is used to run specific model available.
- **data/\***: folder which will contain all *.train* & *.test* files in order to train model.
- **saved_models/*.joblib**: all scikit learn models saved.
- **models_info/***: all markdown files generated to get quick information about model performance and prediction. This folder contains also **model_comparisons.csv** obtained after running runAll_maxwell.sh script.
- **modules/\***: contains all modules usefull for the whole project (such as configuration variables)

### Scripts for generating data files

Two scripts can be used for generating data in order to fit model:
- **generate_data_model.py**: zones are specified and stayed fixed for each scene
- **generate_data_model_random.py**: zones are chosen randomly (just a number of zone is specified)
- **generate_data_model_random_maxwell.py**: zones are chosen randomly (just a number of zone is specified). Only maxwell scene are used.


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
- **sep**: output csv file seperator used
- **rowindex**: if 1 then row will be like that 1:xxxxx, 2:xxxxxx, ..., n:xxxxxx
- **custom**: specify if you want your data normalized using interval and not the whole singular values vector. If it is, the value of this parameter is the output filename which will store the min and max value found. This file will be usefull later to make prediction with model (optional parameter).

### Train model

This is an example of how to train a model

```bash
python train_model.py --data 'data/xxxxx.train' --output 'model_file_to_save' --choice 'model_choice'
```

Expected values for the **choice** parameter are ['svm_model', 'ensemble_model', 'ensemble_model_v2'].

### Predict image using model

Now we have a model trained, we can use it with an image as input:

```bash
python predict_noisy_image_svd.py --image path/to/image.png --interval "x,x" --model saved_models/xxxxxx.joblib --metric 'lab' --mode 'svdn' --custom 'min_max_filename'
```

- **metric**: metric choice need to be one of the listed above.
- **custom**: specify filename with custom min and max from your data interval. This file was generated using **custom** parameter of one of the **generate_data_model\*.py** script (optional parameter).

The model will return only 0 or 1:
- 1 means noisy image is detected.
- 0 means image seem to be not noisy.

All SVD metrics developed need:
- Name added into *metric_choices_labels* global array variable of **modules/utils/config.py** file.
- A specification of how you compute the metric into *get_svd_data* method of **modules/utils/data_type.py** file.

### Predict scene using model

Now we have a model trained, we can use it with an image as input:

```bash
python prediction_scene.py --data path/to/xxxx.csv --model saved_model/xxxx.joblib --output xxxxx --scene xxxx
```
**Remark**: *scene* parameter expected need to be the correct name of the Scene.

### Visualize data

All scripts with names **display_\*.py** are used to display data information or results.

Just use --help option to get more information.

### Simulate model on scene

All scripts named **predict_seuil_expe\*.py** are used to simulate model prediction during rendering process. Do not forget the **custom** parameter filename if necessary.

Once you have simulation done. Checkout your **threshold_map/%MODEL_NAME%/simulation\_curves\_zones\_\*/** folder and use it with help of **display_simulation_curves.py** script.

## Others scripts

### Test model on all scene data

In order to see if a model well generalized, a bash script is available:

```bash
bash testModelByScene.sh '100' '110' 'saved_models/xxxx.joblib' 'svdne' 'lab'
```

Parameters list:
- 1: Begin of interval of data from SVD to use
- 2: End of interval of data from SVD to use
- 3: Model we want to test
- 4: Kind of data input used by trained model
- 5: Metric used by model


### Get treshold map

Main objective of this project is to predict as well as a human the noise perception on a photo realistic image. Human threshold is available from training data. So a script was developed to give the predicted treshold from model and compare predicted treshold from the expected one.

```bash
python predict_seuil_expe.py --interval "x,x" --model 'saved_models/xxxx.joblib' --mode ["svd", "svdn", "svdne"] --metric ['lab', 'mscn', ...] --limit_detection xx --custom 'custom_min_max_filename'
```

Parameters list:
- **model**: mode file saved to use
- **interval**: the interval of data you want to use from SVD vector.
- **mode**: kind of data ['svd', 'svdn', 'svdne']; not normalize, normalize vector only and normalize together.
- **limit_detection**: number of not noisy images found to stop and return threshold (integer).
- **custom**: custom filename where min and max values are stored (optional parameter).

### Display model performance information

Another script was developed to display into Mardown format the performance of a model.

The content will be divised into two parts:
- Predicted performance on all scenes
- Treshold maps obtained from model on each scenes

The previous script need to already have ran to obtain and display treshold maps on this markdown file.

```bash
python save_model_result_in_md.py --interval "xx,xx" --model saved_models/xxxx.joblib --mode ["svd", "svdn", "svdne"] --metric ['lab', 'mscn']
```

Parameters list:
- **model**: mode file saved to use
- **interval**: the interval of data you want to use from SVD vector.
- **mode**: kind of data ['svd', 'svdn', 'svdne']; not normalize, normalize vector only and normalize together.

Markdown file with all information is saved using model name into **models_info** folder.

### Others...

All others bash scripts are used to combine and run multiple model combinations...

## License

[The MIT license](https://github.com/prise-3d/Thesis-NoiseDetection-metrics/blob/master/LICENSE)
