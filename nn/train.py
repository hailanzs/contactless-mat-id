#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
import utility
import argparse
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from model_base import SimpleModel
import processing.dataset as dataset

if __name__ == "__main__":


    # load command line options
    opt = utility.train_options(parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter))

    # experiment name, dataset path, and logging
    exp_name = utility.new_exp(opt.exp_name)
    dataset_path = os.path.join(opt.datapath, '*.mat')
    
    train_dates = utility.parse_dates(opt.train_dates)
    val_dates = utility.parse_dates(opt.val_dates)
    test_dates = utility.parse_dates(opt.test_dates)
    feature_names = utility.parse_features(opt.feature_names)
    loss_weights = utility.parse_loss_weights(opt.loss_weights)
    objects = utility.parse_objects(opt.objects)
    train_for = opt.train_for
    if(train_for == 'Y'):
        output_size = len(objects)
    elif(train_for == 'metal'):
        output_size = 2
    elif(train_for == 'material'):
        output_size = 7
    
    
    dir_path = os.path.dirname(os.path.realpath('README.md'))
    with open(os.path.join(dir_path, 'scripts', 'exp_name.txt'), 'w') as f:
        f.write(exp_name)
    dataroot = os.path.join(dir_path, 'processed_data')
    dir_path = os.path.join(dir_path,'nn')
    # just for us
    dataroot = r"/home/synrg-sc1/Desktop/Sohrab_Mat_Sensing/mat_sensing_network/processed_data"
    if opt.environment == "same":
        train_loader,val_loader,test_loader = dataset.createDataset(dataroot=dataroot,dates=train_dates, input_len=250, 
                                                normalize=False,val_samples=0.1,
                                                cutoff=opt.cutoff, batch_size=opt.batch_size,
                                                sample_limit=opt.sample_limit, train_for = train_for,
                                                feature_names=feature_names, objects_of_interest=objects, lim=(250+30+30))
    else:
        train_loader = dataset.createDataset(dataroot=dataroot,dates=train_dates, input_len=250, 
                                            normalize=False,val_samples=0,
                                            cutoff=opt.cutoff, batch_size=opt.batch_size,
                                            sample_limit=opt.sample_limit, train_for = train_for,
                                            feature_names=feature_names, objects_of_interest=objects, lim=(250+30+30))
        val_loader = dataset.createDataset(dataroot=dataroot,dates=val_dates, input_len=250,
                                        normalize=False,val_samples=0, 
                                        cutoff=opt.cutoff, batch_size=opt.batch_size, 
                                        sample_limit=opt.sample_limit,  train_for = train_for,
                                        feature_names=feature_names, objects_of_interest=objects, lim=30)
        test_loader = dataset.createDataset(dataroot=dataroot,dates=test_dates, input_len=250,
                                        normalize=False,val_samples=0, 
                                        cutoff=opt.cutoff, batch_size=opt.batch_size, 
                                        sample_limit=opt.sample_limit,  train_for = train_for,
                                        feature_names=feature_names, objects_of_interest=objects, lim=30)
    for num_reps in range(opt.reps):
        print("\nstarting new rep: " + str(num_reps))
        # setting manual seed
        if opt.seed > -1:
            torch.manual_seed(opt.seed)
            np.random.seed(opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # define model
        device = utility.get_device()
        print(output_size)
        
        net = SimpleModel(input_size_fft=1, input_size_mrf=25, input_size_phase=3250, 
                          output_size=output_size, fft_channels = int(opt.fft_channels), first_channels=opt.first_channels, p=float(opt.p), device=device)
            
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=3, factor=0.3)
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"created network with {n_params} parameters...")
        # cuda
        net.to(device)
        opt.device = str(device)
        print(device)
        
        for epoch in range(opt.epoch + 1):  
            print(f"\nepoch {epoch}...", end='')

            # variables 
            losses, val_losses, train_losses, train_accu, valid_accu = [], [], [], [], []
            
            # save metadata
            opt.latest_epoch = epoch 
            utility.create_dir(os.path.join(dir_path,"metadata", exp_name + "/" + str(num_reps)))
            utility.save_metadata(exp_name + "/" + str(num_reps) + "/", opt)      

            # train network
            net.train()
            total = 0
            correct = 0
            correct1 = 0
            correct2 = 0
            correct3 = 0
            for i, data in enumerate(train_loader, 0): 
                # load data to device (gpu or cpu)
                X1, X2, X3 = data['X']
                x1, x2, x3, groundtruth, name = (X1).to(device, dtype=torch.float), (X2).to(device, dtype=torch.float), (X3).to(device, dtype=torch.float), data['Y'].to(device, dtype=torch.long), data['name']

                optimizer.zero_grad()
                output, output1, output2, output3 = net(x1, x2, x3)

                loss0, loss1, loss2, loss3 = net.loss(output, groundtruth), net.loss(output1, groundtruth), net.loss(output2, groundtruth), net.loss(output3, groundtruth)
                loss = (loss_weights[0] * loss0 +  loss_weights[1] * loss1 + loss_weights[2] * loss2 + loss_weights[3] * loss3) / np.sum(loss_weights)

                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.detach().cpu().numpy()))

                # log losses and anomalies
                utility.log(output, 'output')
                utility.log(loss, 'loss')
                
                _, predicted = output.max(1)
                total += output.size(0)
                correct += predicted.eq(groundtruth).sum().item()
                accu=100.*correct/total
                train_accu.append(accu)
                
                # print out loss
                if i % opt.print_period == 0:
                        print("\repoch %d... train loss: %.3f ... train acc: %.3f" % (epoch, np.mean(train_losses), np.mean(train_accu)), end='')

            # validation loss  
            net.eval()
            
            total, correct, correct1, correct2, correct3 = 0,0,0,0,0
            wrong_names, wrong_names1, wrong_names2, wrong_names3 = [],[],[],[]
            for i, data in enumerate(val_loader):
                # load data to device (gpu or cpu)
                X1, X2, X3 = data['X']          
                x1, x2, x3, groundtruth, name = (X1).to(device, dtype=torch.float), (X2).to(device, dtype=torch.float), (X3).to(device, dtype=torch.float), data['Y'].to(device, dtype=torch.long), data['name']

                with torch.no_grad():
                    output, output1, output2, output3  = net(x1, x2, x3)
                    loss0, loss1, loss2, loss3 = net.loss(output, groundtruth), net.loss(output1, groundtruth), net.loss(output2, groundtruth), net.loss(output3, groundtruth)
                    loss = (loss_weights[0] * loss0 +  loss_weights[1] * loss1 + loss_weights[2] * loss2 + loss_weights[3] * loss3) / np.sum(loss_weights)

                    val_losses.append(float(loss.detach().cpu().numpy()))
                    
                    _, predicted = output.max(1)
                    total += output.size(0)
                    correct += predicted.eq(groundtruth).sum().item()
                    accu=100.*correct/total
                    valid_accu.append(accu)
                    wn = np.where(predicted.eq(groundtruth).cpu() == False)[0]
                    if(len(wn) > 0):
                        wrong_names.append([name[wn[i]] for i in range(len(wn))])
                    
                    
                    # print out loss
                    if i % opt.print_period == 0:
                        print("\repoch %d... train loss: %.3f val loss: %.3f...train accu: %.3f val accu: %.3f" % (epoch, np.mean(train_losses), np.mean(val_losses), np.mean(train_accu), np.mean(valid_accu)), end='')

                    
            # step scheduler
            scheduler.step(np.mean(val_losses))

            # save model
            if epoch % opt.save_period == 0:
                utility.create_dir(os.path.join(dir_path,"checkpoints", exp_name, str(num_reps)))
                utility.save_model(net, exp_name + "/" + str(num_reps), epoch)
            utility.create_dir(os.path.join(dir_path,"logs", exp_name))
            utility.create_dir(os.path.join(dir_path,"logs", exp_name, str(num_reps)))
            wrong_names = [val for sublist in wrong_names for val in sublist]
            sio.savemat(os.path.join(dir_path,"logs/" + exp_name + "/" + str(num_reps) + '/' + str(epoch) + ".mat"), {'t_loss': train_losses, 'v_loss': val_losses, 't_accu': train_accu, 'v_accu': valid_accu,'wrong_names': wrong_names} )

            # test on validation dataset 
            test_path = os.path.join(dir_path,"results", exp_name)
            utility.create_dir(test_path)
            test_path = os.path.join(dir_path,"results", exp_name , str(num_reps))
            utility.create_dir(test_path)
            net.eval()
            for ii, data in enumerate(test_loader):
                # load data to device (gpu or cpu)
                
                X1, X2, X3 = data['X']
                x1, x2, x3, groundtruth, name = (X1).to(device, dtype=torch.float), (X2).to(device, dtype=torch.float), (X3).to(device, dtype=torch.float), data['Y'].to(device, dtype=torch.long), data['name']

                with torch.no_grad():
                    output = net(x1, x2, x3)

                    sio.savemat(test_path + "/" + str(ii) + ".mat", mdict={
                        'x1':x1.detach().cpu().numpy(),
                        'x2':x2.detach().cpu().numpy(),
                        'x3':x3.detach().cpu().numpy(),
                        # 'output':output.detach().cpu().numpy(),
                        'output':output,
                        'gt':groundtruth.detach().cpu().numpy(),
                        'name': data['name'],
                        'objects': data['obj']
                    })