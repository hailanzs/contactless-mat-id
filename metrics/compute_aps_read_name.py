import os
import scipy.io as sio
import numpy as np
from sklearn.metrics import average_precision_score
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import nn.utility as utility
from scipy.special import softmax
import argparse


dir_path = os.path.dirname(os.path.realpath('README.md'))

# experiment name, dataset path, and logging
with open(os.path.join(dir_path, 'scripts', 'exp_name.txt'), 'r') as f:
    exp_name = str(f.readlines())[2:-2]
print(exp_name)
    
opt = utility.view_metrics_options(parser=argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter))
dir_path = os.path.join(dir_path, 'nn')
path_to_use = os.path.join(dir_path, 'results')
all_aps, all_accs = [], []
metadata_path = os.path.join(dir_path,"metadata", exp_name, str(0), ".mat")
opt_loaded = sio.loadmat(metadata_path)['opt']

if opt.tested_results == 0:
    path_to_use = os.path.join(dir_path, 'tested_results')
    
objects = utility.parse_objects(str(opt_loaded['objects'][0][0][0]))
train_for = opt_loaded['train_for'][0][0][0]
for exp_rep in range(opt_loaded['reps'][0][0][0][0]):
    if(train_for == 'Y'):
        output_size = len(objects)
    elif(train_for == 'metal'):
        output_size = 2
        objects = ['metal', 'non-metal']
    elif(train_for == 'material'):
        output_size = 7
        objects = ["cardboard", "ceramic", "foam", "glass", "wood", "metals", "plastic"]
    n_obj = output_size

    con_mat= np.zeros(shape=(n_obj, n_obj))
    scores,  test_date, true_labels = [], [], []
    for filename in range(50000):
        filepath = os.path.join(path_to_use, exp_name, str(exp_rep), str(filename))
        if(os.path.isfile(filepath + '.mat')): 
            A = sio.loadmat(filepath)
            x1 = A['x1']
            output = A['output']
            name = A['name']
            gt = A['gt']
            batch_size = x1.shape[0]
            for batch in range(batch_size):
                    
                y = np.argmax(output[batch,:])
                
                reind_gt = int(gt[0][batch])
                con_mat[reind_gt, int(y)] += 1
                scores += [list(output[batch, :])]
                true_label = np.zeros(shape=(n_obj,))
                true_label[reind_gt] = 1
                true_labels += [list(true_label)]

    # calculate AP
    ap = average_precision_score(true_labels, [softmax(score) for score in scores], average=None)
    print("Iteration: " + str(exp_rep))
    print("average precision:")
    for ii, obj in enumerate(objects):
        print("%30s : full: %.3f" % (obj, ap[ii]))

    print("%30s : full: %.3f" % ("MEAN AP", np.mean(ap)))    
    total = np.sum(con_mat,axis=1)
    print(con_mat)
    acc = np.diag(con_mat) / total
    
    overall_acc = np.mean(acc)

    print("average Accuracy:")
    for ii, obj in enumerate(objects):
        print("%30s : full: %.3f" % (obj, acc[ii]))

    print("%30s : full: %.3f" % ("OVERALL ACC", overall_acc))

    all_aps += [ap]
    
    all_accs += [acc]
    
all_aps = np.asarray(all_aps)

stds = np.std(all_aps, axis=0)

axis1_all_aps = np.mean(all_aps, axis=1)

axis0_all_aps = np.mean(all_aps, axis=0)

print("\n\n Final Average Precision:")
for ii, obj in enumerate(objects):
        print("%30s : full: %7.3f/%6.3f" % (obj, axis0_all_aps[ii]*100, stds[ii]*100))
        
print("%30s : full: %7.3f/%6.3f" % ("Mean all_ap", np.mean(axis1_all_aps)*100, np.std(axis1_all_aps)*100))

all_accs = np.asarray(all_accs)

stds = np.std(all_accs, axis=0)

axis1_all_accs = np.mean(all_accs, axis=1)

axis0_all_accs = np.mean(all_accs, axis=0)

print(axis1_all_accs)
print(axis0_all_accs)


print("\n\n Final Average Accuracy:")
for ii, obj in enumerate(objects):
        print("%30s : full: %7.3f/%6.3f" % (obj, axis0_all_accs[ii]*100, stds[ii]*100))
        
print("%30s : full: %7.3f/%6.3f" % ("Mean all_acc", np.mean(axis1_all_accs)*100,np.std(axis1_all_accs)*100))



