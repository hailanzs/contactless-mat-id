import torch
import numpy as np

import dataset_idx as dataset
import scipy.io as sio
import os
import utility
import argparse
import torch.nn.functional as nnf


def get_checkpoint_path(exp, epoch):
    dir_path = os.path.dirname(os.path.realpath('nn/train.py'))
    return os.path.join(dir_path,"checkpoints", str(exp), "model_" + str(epoch) + ".pt")

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath('nn/train.py'))
    
    device = utility.get_device()
    opt = utility.test_options(parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter))

    meta_path = os.path.join(dir_path,'metadata/', opt.exp_name + "/0/.mat")
    if os.path.exists(meta_path):
        metadata_file = sio.loadmat(meta_path)
        
        # these 3 paramterse are not overwritten:
        exp_to_load = opt.exp_name
        epoch_to_load = opt.epoch
        dates = utility.parse_dates(opt.test_dates)

        loaded_opt = metadata_file['opt']
    else:
        exp_to_load = opt.exp_name
        epoch_to_load = opt.epoch
        dates = utility.parse_dates(opt.test_dates)

    objects = utility.parse_objects(opt.objects)
    for rep in range(10):
        checkpoint_path = get_checkpoint_path(exp_to_load + "/" + str(rep) , epoch_to_load)
        model = torch.load(checkpoint_path, map_location=device)
        model.eval()

        result_path = os.path.join(dir_path, "results", exp_to_load +"_epoch_" + str(epoch_to_load) + "_" + dates[0])
        utility.create_dir(result_path)
        result_path = os.path.join(os.path.join(result_path, str(rep)))
        print(dates)
        utility.create_dir(result_path)
        feature_names = utility.parse_features(loaded_opt['feature_names'][0][0][0])

        test_loader = dataset.createDataset(dates=dates, input_len=250, 
                                            normalize=False,val_samples=0, objects_of_interest=objects, train_for = opt.train_for, 
                                            cutoff=opt.cutoff, batch_size=opt.batch_size, sample_limit=opt.sample_limit, feature_names=feature_names)
        model.eval()
        for ii, data in enumerate(test_loader):
            # load data to device (gpu or cpu)
            
            X1, X2, X3 = data['X']
            x1, x2, x3, groundtruth, name = (X1).to(device, dtype=torch.float), (X2).to(device, dtype=torch.float), (X3).to(device, dtype=torch.float), data['Y'].to(device, dtype=torch.long), data['name']
                    
            output, output1, output2, output3 = model(x1, x2, x3)
            sio.savemat(result_path + "/" + str(ii) + ".mat", mdict={
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
