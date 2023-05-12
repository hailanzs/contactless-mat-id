# Contactless Material Identification with Millimeter Wave Vibrometry
This repository contains the training, processing and data capture code for RFVibe. More detailed installation and running instructions are detailed below.

## Requirements
- Computer with GPU that runs Cuda
- at leaset 50 GB of space for dataset
## Software Installation Instructions
1. Clone and install the repository:
```
git clone git@github.com:hshanbha/contactless-mat-id.git
```
2. Install the required python packages:
```
cd contactless-material-id
pip install -r requirements.txt
```
3. Download dataset from [this Google Drive](https://drive.google.com/drive/folders/1QHKSPK9nfHmPVjcHXf1ZVb9CegpqqyEz?usp=sharing), unzip and place in *contactless-material-id/*.
## Training
In order to train with the default commands for the main results, change permissions for the run script (needs to be done once):
```
chmod +x scripts/run_command.bash
chmod +x scripts/test_command.bash
```
Then run the bash script to run both material-wise classification and object-wise classification. 
```
./scripts/run_command.sh
```
The default arguments are placed in utility.py. 

If you would like to change any paramters, the listed arguments are in utility in the function train_options(). Simply run the command:
```
python nn/train.py --<option_listed> <value_given>
```
For example, to change the dataset split or the objects to train and test on, you may run:
```
python nn/train.py --environment different 
python nn/train.py --objects "aluminum, brass, copper, steel"
```

## Testing
If you would like to only test a given dataset on an already trained model, run train.py with the experiment name created by train.py (can be found in results/, metadata/ or logs/) and the folder to the new experiments to test. More arguments can be found in utility.py in the function test_options().
```
python nn/test.py --exp_name <exp_name_to_test> --test_dates <new_test_dates>
```
In addition, we have added pre-trained models to evaluate as needed in nn/checkpoints. Bash scripts are provided to automatically run the main results and view metrics:
```
./scripts/test_command.sh
```

## Viewing Results
Copy and paste the new experiment name that is listed in the updated folders (can check any of metadata, results, logs and checkpoints folders), and input that as an argument for *processing/compute_aps.py*. You can also specify whether to view the results from when the model is evaluated by the test dataset that is specified in nn/train.py or the results from the test dataset run in nn/test.py. The rest of the arguments can be viewed in nn/utility.py in the view_metrics_options() function.
```
python metrics/compute_aps.py --exp_name <copied_experiment_name> --tested_results <1 or 0 >
```
For example, 
```
python metrics/compute_aps.py --exp_name artifact_eval_2023_04_04-02_52_23
```
The averaged accuracy will be printed to the terminal for each catergory, depending on the training mode (objects, materials).

## Expected Results
| Experiment                             | Acc  | Location                              |
| -------------------------------------- | ---- | --------------------------------------|
| Material-wise, Similar Environments    | 83%  | nn/checkpoints/main_material          |
| Material-wise, Different Environments  | 71%  | nn/checkpoints/main_material_different|
| Object-wise, Similar Environments      | 74%  | nn/checkpoints/main_objs              |
| Object-wise, Different Environments    | 68%  | nn/checkpoints/main_objs_different    |

## Data Capture and Processing Pipeline
Here we describe the process for capturing new experiments to run RFVibe's network on. The experiment pipeline is as follow: setup (hardware and software), data collection, and post-processing.

### Hardware Installation Instructions
The following pieces of hardware are required to replicate our experiments:
- Klipsch Sub100 Subwoofer
- TI's XWR1843BOOST Evaluation Board
- DCA1000 Data Capture Board
- Three separate surfaces (for the mmWave board, object of interest, and speaker)

### Software Installation and Instructions
Assuming git@github.com:hshanbha/contactless-mat-id.git is already installed.
1. Clone and install the OpenRadar repository inside *contactless-material-id/data-capture/*:
```
git clone -b adc-fixes https://github.com/presenseradar/openradar
```
2. Install the required python packages:
```
cd openradar
pip install -r requirements.txt
```

### Experiment Set Up
In order to set up the experiments, connect the 1843BOOST Evaluation Board and DCA1000 as instructed on TI's user guide. In addition, download and install mmWave Studio to program the mmWave device. 

1. Connect the mmWave device and speaker to the same computer. Set up the experiments similar to the setups shown in the picture below. 

![plot](https://github.com/hshanbha/contactless-mat-id/blob/main/images/setup.png?raw=true)

2. Turn on mmWave Studio and run *contactless-material-id/data-capture/run.lua*. Make sure to change the hardcoded COM port to match the one of the system. 
3. Once the radar is capture data, exit the mmWave Studio window and open Task Manager and kill the process starting with DCA1000. 
4. Open *contactless-material-id/data-capture/prod-cons-capture.lua*, under the main function, change the number of **iterations** to the desired number, change the **exp_object** to the label for the current object, and **date** to the folder name to save the current set of experiments. 
5. Run *contactless-material-id/data-capture/prod-cons-capture.lua* to capture the data.

### Data Processing Pipeline
(Optionally, but recommened): Calibrate the XWR1843BOOST radar and input the calibration matrix to *contactless-material-id/data-capture/save_calib_data.m*
1. Add the date to the top of *contactless-material-id/data-capture/save_calib_data.m*
2. Run *contactless-material-id/data-capture/save_calib_data.m* to calibrate data and reformat into channels. 

Note: To process the new data, make sure to add the folder name to the list of train, validation or test dates listed in *utility.py*.
