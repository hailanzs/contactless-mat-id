# Contactless Material Identification with Millimeter Wave Vibrometry
This is the processing and training code for RFVibe. 
## Requirements
- Computer with GPU that runs Cuda
- at leaset 50 GB of space for dataset
## Software Installation Instructions
1. Clone and install the repository:
```
git clone https://gitlab.engr.illinois.edu/hshanbha/contactless-material-id.git
```
2. Install the required python packages:
```
cd contactless-material-id
pip install -r requirements.txt
```
3. Download dataset from [this Google Drive](https://drive.google.com/drive/folders/1QHKSPK9nfHmPVjcHXf1ZVb9CegpqqyEz?usp=sharing), unzip and place in *contactless-material-id/*.
## Training
In order to train with the default commands, run:
```
./scripts/run_command.bash
```
The default arguments are placed in utility.py. 

If you would like to change any paramters, the listed arguments are in utility in the function train_options(). Simply run the command:
```
python nn/train.py --<option_listed> <value_given>
```
For example, to change the dataset split you may run:
```
python nn/train.py --environment different
```

## Viewing Results

## (Extra) Data Capture and Processing Pipeline
This is the data capture and processing for the experiments. 

### Hardware Installation Instructions
The following pieces of hardware are required to replicate our experiments:
- Klipsch Sub100 Subwoofer
- TI's 1843BOOST Evaluation Board
- DCA1000 Data Capture Board
- Three separate surfaces (for the mmWave board, object of interest, and speaker)

### Software Installation and Instructions
Assuming https://gitlab.engr.illinois.edu/hshanbha/contactless-material-id.git is already installed.
1. Clone and install the OpenRadar repository inside *contactless-material-id/data-capture/*:
```
git clone -b adc-fixes https://github.com/presenseradar/openradar
```
2. Install the required python packages:
```
cd openradar
pip install -r requirements.txt
```

### Experiment Set up
In order to set up the experiments, connect the 1843BOOST Evaluation Board and DCA1000 as instructed on TI's user guide. In addition, download and install mmWave Studio to program the mmWave device. 

1. Connect the mmWave device and speaker to the same computer. Set up the experiments as shown in the picture below. 

![alt text](https://github.com/hshanbha/contactless-mat-id/blob/main/setup.png?raw=true)

2. Turn on mmWave Studio and run *contactless-material-id/data-capture/run.lua*. Make sure to change the hardcoded COM port to match the one of the system. 
3. Once the radar is capture data, exit the mmWave Studio window and open Task Manager and kill the process starting with DCA1000. 
4. Open *contactless-material-id/data-capture/prod-cons-capture.lua*, under the main function, change the number of **iterations** to the desired number, change the **exp_object** to the label for the current object, and **date** to the folder name to save the current set of experiments. 
5. Run *contactless-material-id/data-capture/prod-cons-capture.lua* to capture the data.

### Data Processing Pipeline
(Optionally, but recommened): Calibrate the XWR1843BOOST radar and input the calibration matrix to *contactless-material-id/data-capture/save_calib_data.m*
1. Add the date to the top of *contactless-material-id/data-capture/save_calib_data.m*
2. Run *contactless-material-id/data-capture/save_calib_data.m*.

