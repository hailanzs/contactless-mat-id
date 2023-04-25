import torch
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import processing.dataset as dataset
import scipy.io as sio
import utility
import argparse


def get_checkpoint_path(exp, epoch):
    dir_path = os.path.dirname(os.path.realpath('nn/train.py'))
    return os.path.join(dir_path,"checkpoints", str(exp), "model_" + str(epoch) + ".pt")

if __name__ == "__main__":
    
    dir_path = os.path.dirname(os.path.realpath('README.md'))
    dataroot = os.path.join(dir_path, 'processed_data')
    dir_path = os.path.join(dir_path,'nn')
    device = utility.get_device()
    opt = utility.test_options(parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter))
    dataroot = r"/home/synrg-sc1/Desktop/Sohrab_Mat_Sensing/mat_sensing_network/processed_data"
    exp_name = utility.new_exp(opt.exp_name)

    meta_path = os.path.join(dir_path,'metadata/', opt.exp_name + "/0/.mat")
    if os.path.exists(meta_path):
        metadata_file = sio.loadmat(meta_path)
        loaded_opt = metadata_file['opt']
        
        # these 3 paramterse are not overwritten:
        exp_to_load = opt.exp_name
        epoch_to_load = opt.epoch
        if opt.load_dates == 0:
            dates = utility.parse_dates(metadata_file['opt']['test_dates'][0][0][0])
        else:
            dates = utility.parse_dates(opt.test_dates)

    else:
        exp_to_load = opt.exp_name
        epoch_to_load = opt.epoch
        dates = utility.parse_dates(opt.test_dates)

    objects = utility.parse_objects(metadata_file['opt']['objects'][0][0][0])
    if(metadata_file['opt']['train_for'][0][0][0] == 'Y'):
        output_size = len(objects)
    elif(metadata_file['opt']['train_for'][0][0][0] == 'metal'):
        output_size = 2
    elif(metadata_file['opt']['train_for'][0][0][0] == 'material'):
        output_size = 7
    print(dir_path)

    result_path = os.path.join(dir_path, "tested_results", exp_to_load)
    utility.create_dir(result_path)
    feature_names = utility.parse_features(loaded_opt['feature_names'][0][0][0])
    test_loader = dataset.createDataset(dataroot=dataroot,dates=dates, input_len=250,
                                    normalize=False,val_samples=[0], 
                                    cutoff=metadata_file['opt']['cutoff'][0][0][0][0], batch_size=int(metadata_file['opt']['batch_size'][0][0][0][0]),
                                    sample_limit=10,  train_for = metadata_file['opt']['train_for'][0][0][0],
                                    feature_names=feature_names, objects_of_interest=objects)
    for rep in range(loaded_opt['reps'][0][0][0][0]):
        checkpoint_path = get_checkpoint_path(exp_to_load + "/" + str(rep) , epoch_to_load)
        model = torch.load(checkpoint_path, map_location=device)
        model.eval()
        result_path_sub = os.path.join(os.path.join(result_path, str(rep)))
        utility.create_dir(result_path_sub)
        print(dates)
        
        model.eval()
        for ii, data in enumerate(test_loader):
            # load data to device (gpu or cpu)
            
            X1, X2, X3 = data['X']
            x1, x2, x3, groundtruth, name = (X1).to(device, dtype=torch.float), (X2).to(device, dtype=torch.float), (X3).to(device, dtype=torch.float), data['Y'].to(device, dtype=torch.long), data['name']
                    
            output, output1, output2, output3 = model(x1, x2, x3)
            sio.savemat(result_path_sub + "/" + str(ii) + ".mat", mdict={
                            'x1':x1.detach().cpu().numpy(),
                            'x2':x2.detach().cpu().numpy(),
                            'x3':x3.detach().cpu().numpy(),
                            'output':output.detach().cpu().numpy(),
                            'output1':output1.detach().cpu().numpy(),
                            'output2':output2.detach().cpu().numpy(),
                            'output3':output3.detach().cpu().numpy(),
                            'gt':groundtruth.detach().cpu().numpy(),
                            'name': data['name'],
                            'objects': data['obj']
                        })
            print(f"ran batch {ii}...", end='\r')
