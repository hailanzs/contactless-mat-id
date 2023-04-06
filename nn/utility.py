import os
import numpy as np
import logging
import torch
import time
import argparse
import scipy.io as sio


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def create_dir(this_path):
    if not os.path.isdir(this_path):
        os.mkdir(this_path)   

def new_exp(name):
    dir_path = os.path.dirname(os.path.realpath('nn/train.py'))
    exp_name = str(name + time.strftime("_%Y_%m_%d-%H_%M_%S"))
    create_dir(os.path.join(dir_path,"checkpoints"))
    create_dir(os.path.join(dir_path,"checkpoints",exp_name))
    create_dir(os.path.join(dir_path, "logs"))
    remove_file(os.path.join(dir_path,"logs", exp_name + '.log'))
    create_dir(os.path.join(dir_path,"results"))
    create_dir(os.path.join(dir_path,"results", exp_name))
    create_dir(os.path.join(dir_path,"metadata"))
    create_dir(os.path.join(dir_path,"metadata", exp_name))

    return exp_name

def check_nan(X):
    return np.isnan(X.detach().cpu().numpy()).any()

def log(X, name):
    isnan = check_nan(X)
    if isnan:
        logging.warning(name)

def log_input(x):
    isnan = check_nan(x)
    if isnan:
        logging.warning('input')
    else:
        logging.info("input size: %s" % str(x.size()))

def save_model(net, exp_name, epoch):
    dir_path = os.path.dirname(os.path.realpath('nn/train.py'))
    this_path = os.path.join(dir_path, "checkpoints", exp_name,"model_" + str(epoch) + ".pt")
    torch.save(net, this_path)

def remove_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def print_options(opt):

    print("="*50, )
    print("options:")
    for option, value in opt.__dict__.items():
        print(option.ljust(20, '-'), str(value).rjust(20, '-'))
    print("="*50)


def test_options(parser):
    parser.add_argument("--exp_name", type=str, default='angle-benchmark_2022_12_06-13_22_44', help="experiment to load")
    parser.add_argument("--epoch", type=int, default=30, help="epoch to load")
    parser.add_argument("--skip", type=str, default="NOT_IN_TRAINING", help="filename to skip during training")
    parser.add_argument("--datapath", type=str, default="", help="path to dataset used to test")

    parser.add_argument("--test_dates", type=str, default="dec-6-angle-7,dec-6-angle-6,dec-6-angle-5,dec-6-angle-4,dec-6-angle-3,dec-6-angle-2,dec-6-angle-1,dec-6-2-angle-7,dec-6-2-angle-6,dec-6-2-angle-5,dec-6-2-angle-4,dec-6-2-angle-3,dec-6-2-angle-2,dec-6-2-angle-1,", help="comma seperated train dates") # , feb-10-2, feb-10-3, feb-10-4, feb-11, feb-11-1
    parser.add_argument("--sample_limit", type=int, default=10, help="limit to the number of samlpes from each date")
    parser.add_argument("--feature_names", type=str, default="fft, mrf_squared, damp", help="comma separated feature names: 'fft' | 'pwelch' | 'stft' | 'mrf' | 'mrf_squared'")
    parser.add_argument("--exp_detail", type=str, default="anles", help="comma separated feature names: 'fft' | 'pwelch' | 'stft' | 'mrf' | 'mrf_squared'")
    parser.add_argument("--objects", type=str, default= "aluminum, brass, copper, steel", help="objects of interest we wanna look at")
    parser.add_argument("--iter", type=str, default="all", help="objects of interest we wanna look at")
    parser.add_argument("--train_for", type=str, default="Y", help="Y, material, metal")

    # network parameters
    parser.add_argument("--n1", type=int, default=3, help="number of layers before self-similarity")
    parser.add_argument("--n2", type=int, default=3, help="number of layers after self-similarity")
    parser.add_argument("--seed", type=int, default=0, help="manual seed for reproducibility | -1 means no manual seed")  
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")  
    parser.add_argument("--max_pool", action="store_true", help="whether or not layers before SSM include max-pooling")
    parser.add_argument("--euclidean", action="store_true", help=" use euclidean distance for self-similarity matrix (default: correlation)")
    parser.add_argument("--cutoff", type=int, default=10, help="indices to cut off from the beginnig of FFT")
    parser.add_argument("--fft_channels", type=int, default=5, help="fft channels for the NN (depends on first feature)")

    # dataset parameters
    parser.add_argument("--samp_len", type=int, default=1536, help="# samples in each segment (every 500 is 1 second)")  
    parser.add_argument("--output_size", type=int, default=64, help="size of output (higher means higher resolution)")
    parser.add_argument("--low_noise", type=float, default=0, help="std of gaussian noise added to all of the signals everywhere")
    parser.add_argument("--high_noise", type=float, default=0, help="std of gaussian noise added to some parts of the signal")
    parser.add_argument("--expand", action='store_true', help="whether or not expand or squeeze the data (augmentation)")    
    parser.add_argument("--offset", action='store_true', help="whether or not create random offsets to segments (augmentation)")
    parser.add_argument("--change_shape", action='store_true', help="change the shape of the training dataset randomly each time")
    
    opt = parser.parse_known_args()[0]
    print_options(opt)
    return opt

def train_options(parser):
    parser.add_argument("--exp_name", type=str, default='artifact_eval', help="experiment to load")
    parser.add_argument("--skip", type=str, default="NOT_IN_TRAINING", help="filename to skip during training")
    parser.add_argument("--datapath", type=str, default="", help="path to dataset used to train")
    parser.add_argument("--train_dates", type=str, default="mar-8, mar-8-2, mar-15, mar-15-3, mar-15-6, mar-18-3, mar-3-2, mar-3-5, mar-3-7, feb-10-2, feb-10-3, feb-10-4, feb-11, feb-11-1, feb-11-2, feb-22-1, feb-23, feb-23-1, mar-15-5, mar-16-1, mar-16-2, mar-16-4, mar-18-6, mar-2-2, mar-2-3, mar-22, mar-22-1, mar-22-2, mar-22-6, mar-23-1, mar-23-2,  mar-23-4, mar-23-5,  mar-3-6, mar-3-9, mar-4, feb-23-2, feb-25, mar-15-4, mar-16-5, mar-16-6, mar-16-7, mar-16-8, mar-16-9, mar-17-2, mar-17-3, mar-17-5, mar-17-6, mar-18-2, mar-22-3, mar-23, mar-23-3, apr-26, apr-29,  may-3, may-9, may-11, may-11-1, may-11-2, may-11-3, june-11,june-11-1,june-11-2,june-11-3,june-12-1,june-12-2,june-14-1,may-19-metals2,may-20-1,may-20-2,may-20-3,may-20-4,may-23-1,may-23-2,may-23-3,may-23-4,may-23-5,may-24-1,may-24-2,may-24-3,may-25-1,may-25-2,may-25-3,may-26,may-26-1,may-26-2,may-26-3,may-26-4,may-27-1,may-27-6,may-27-7,may-30-1,may-30-2,may-31-2,may-31-3,may-31-4,june-1,june-1-2,june-1-3,june-1-4,june-2-3,june-2-6,june-2-7,june-3-1,june-3-2,june-3-3,june-3-4,june-5-1,june-7-4,june-7-5,june-8-1,june-8-2,june-9-1,june-9-2,june-9-3,june-10-1,june-10-2,june-10-3,june-10-4,june-13-1,june-13-2,june-13-3,may-14-1,may-14-2,may-15,may-15-1,may-15-2,may-15-3,may-15-4,may-15-5,may-16,may-16-2,may-16-3,may-18", help="comma seperated train dates") 
    parser.add_argument("--val_dates", type=str, default='mar-17, mar-17-1, june-1-1,june-2-4,june-6-2,june-6-3,june-6-4,june-7-1,june-7-2,june-7-3, mar-4-1,mar-7-2, mar-7-3,may-27-3,may-27-4,may-27-5, mar-22-2, mar-23-2, may-30-3, may-30-4', help="comma seperated validation dates")
    parser.add_argument("--test_dates", type=str, default='mar-17-7,mar-17-4, mar-18, mar-18-1, mar-18-4, mar-18-7, mar-2-4, mar-22-4, mar-22-5,  may-10, may-11-4, may-11-5, may-13, may-14, may-16-1, may-27-2, may-31-1,june-13-4,jun-13-5,june-13-6,june-14,june-14-2,june-14-3,june-14-4', help="comma seperated test dates")
    parser.add_argument("--environment", type=str, default='different', help="train/test on same or different environment")
    parser.add_argument("--fft_channels", type=int, default=5, help="number of fft channels, will depend on first feature")
    
    parser.add_argument("--sample_limit", type=int, default=10000, help="limit to the number of samlpes from each date")
    parser.add_argument("--feature_names", type=str, default="fft, mrf_squared, damp", help="comma separated feature names: 'fft' | 'pwelch' | 'stft' | 'mrf' | 'mrf_squared'")
    parser.add_argument("--objects", type=str, default='aluminum, usrp, brass, candle-jar, cardboard, ceramic, ceramic-bowl, clorox-wipes, copper, foam, foam-head, febreeze, glass-food-container, hardwood, metal-box, metal-pot, plastic, plastic-box, plastic-food-container, steel, trash-bin, wine-glass, wood', help="objects of interest we wanna look at") 
    parser.add_argument("--test_objects", type=str, default='', help="objects of interest we wanna look at") 
    parser.add_argument("--epoch", type=int, default=30, help="# epochs to run")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay L2 penalty")
    parser.add_argument("--train_for", type=str, default="material", help="Y, material, metal")
    
    # network parameters
    parser.add_argument("--input_size", type=int, default=3250, help="length of the 1D input to the network")
    parser.add_argument("--seed", type=int, default=-1, help="manual seed for reproducibility | -1 means no manual seed")  
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")  
    parser.add_argument("--cutoff", type=int, default=10, help="indices to cut off from the beginnig of FFT")
    parser.add_argument("--first_channels", type=int, default=8, help="number of channels after the first layer")
    parser.add_argument("--loss_weights", type=str, default="1,0.9,0.3,0.3", help="weights to associate with each feautures loss function and overall loss function")
    parser.add_argument("--p", type=str, default="0.25", help="drop out probability")

    # misc
    parser.add_argument("--save_period", type=int, default=30, help="save model every n epochs")  
    parser.add_argument("--log", action="store_true", help="whether or not log this run")
    parser.add_argument("--print_period", type=int, default=10, help="print loss after every n steps")
    opt = parser.parse_known_args()[0]

    print_options(opt)
    return opt

def view_metrics_options(parser):
    parser.add_argument("--exp_name", type=str, default='angle-benchmark_2022_12_06-13_22_44', help="experiment to load")
    
    opt = parser.parse_known_args()[0]
    print_options(opt)
    return opt

def get_device():
    return torch.device("cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu")

def remove_empty_folders(root):
    folders = list(os.walk(root))[1:]
    for folder in folders:
        if not folder[2]:
            os.rmdir(folder[0])

def remove_empty_files(root):
    files = list(os.walk(root))[0][2]

    for this_file in files:
        filepath = root + "/" + this_file
        if os.stat(filepath).st_size == 0:
            os.remove(filepath)

def save_metadata(exp_name, opt):
    dir_path = os.path.dirname(os.path.realpath('nn/train.py'))
    sio.savemat(os.path.join(dir_path,"metadata", exp_name + ".mat"), {"opt": opt})
    
    
def load_processed_data(exp_date, exp_name, root_path='/media/synrg-sc1/Seagate_Expansion_Drive/Sohrab_Mat_Sensing/mat_sensing_network'):
    data_dict = sio.loadmat(os.path.join(root_path, exp_date,  str(exp_name)))
    return data_dict

def save_processed_data(exp_date, exp_name, data, exp_object,folder_name):
    exp_path = os.path.join(folder_name, exp_date, str(exp_name) + ".mat")
    sio.savemat(exp_path, {"data": data, "exp_object": exp_object, "exp_num": exp_name})


def parse_dates(dates):
    A = dates.split(',')
    return [x.strip() for x in A]


def parse_features(features):
    A = features.split(',')
    return [x.strip() for x in A]

def parse_objects(objects):
    A = objects.split(',')
    return [x.strip() for x in A]


def parse_loss_weights(loss_weights):
    A = loss_weights.split(',')
    return [float(x.strip()) for x in A]
    
def split_dates_dataset(fast_dataset, lengths):
    t_len = lengths[0]
    v_len = lengths[1]

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        
        
        
        
        
        
        
        
