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
opt = utility.view_train_options(parser=argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter))

# experiment name, dataset path, and logging
with open(os.path.join(dir_path, 'scripts', 'exp_name.txt'), 'r') as f:
    exp_name = str(f.readlines())[2:-2]
print(exp_name)
    
dir_path = os.path.join(dir_path, 'nn')
path_to_use = os.path.join(dir_path, 'results')
all_aps, all_aps1, all_aps2, all_aps3, all_accs, all_accs1, all_accs2, all_accs3, = [], [], [], [], [], [], [], []
for exp_rep in range(1):
    metadata_path = os.path.join(dir_path,"metadata", exp_name, str(exp_rep), ".mat")
    opt = sio.loadmat(metadata_path)['opt']
    objects = utility.parse_objects(str(opt[0][0][8][0]))
    train_for = opt.train_for
    if(train_for == 'Y'):
        output_size = len(objects)
    elif(train_for == 'metal'):
        output_size = 2
    elif(train_for == 'material'):
        output_size = 7
    n_obj = output_size

    con_mat, con_mat1, con_mat2, con_mat3= np.zeros(shape=(n_obj, n_obj)), np.zeros(shape=(n_obj, n_obj)), np.zeros(shape=(n_obj, n_obj)), np.zeros(shape=(n_obj, n_obj))
    scores, scores1, scores2, scores3, test_date, true_labels = [], [], [], [], [], []
    for filename in range(50000):
        filepath = os.path.join(path_to_use, exp_name, str(exp_rep), str(filename))
        if(os.path.isfile(filepath + '.mat')): 
            A = sio.loadmat(filepath)
            x1 = A['x1']
            output = A['output']
            name = A['name']
            output1 = A['output1']
            output2 = A['output2']
            output3 = A['output3']
            gt = A['gt']
            batch_size = x1.shape[0]
            for batch in range(batch_size):
                if True:
                    
                    y = np.argmax(output[batch,:])
                    y1 = np.argmax(output1[batch,:])
                    y2 = np.argmax(output2[batch,:])
                    y3 = np.argmax(output3[batch,:])
                    
                    reind_gt = int(gt[0][batch])
                    con_mat[reind_gt, int(y)] += 1
                    con_mat1[reind_gt, int(y1)] += 1
                    con_mat2[reind_gt, int(y2)] += 1
                    con_mat3[reind_gt, int(y3)] += 1
                    
                    scores1 += [list(output1[batch, :])]
                    scores2 += [list(output2[batch, :])]
                    scores3 += [list(output3[batch, :])]
                    scores += [list(output[batch, :])]
                    true_label = np.zeros(shape=(n_obj,))
                    true_label[reind_gt] = 1
                    true_labels += [list(true_label)]

    ap = average_precision_score(true_labels, [softmax(score) for score in scores], average=None)
    ap1 = average_precision_score(true_labels, scores1, average=None)
    ap2 = average_precision_score(true_labels, scores2, average=None)
    ap3 = average_precision_score(true_labels, scores3, average=None)
    print("Iteration: " + str(exp_rep))
    print("average precision:")
    for ii, obj in enumerate(objects):
        print("%30s : full: %.3f, fft: %.3f, mrf: %.3f, phase: %.3f " % (obj, ap[ii], ap1[ii], ap2[ii], ap3[ii]))

    print("%30s : full: %.3f, fft: %.3f, mrf: %.3f, phase: %.3f " % ("MEAN AP", np.mean(ap),
                                                                    np.mean(ap1), np.mean(ap2), np.mean(ap3)))    
    total = np.sum(con_mat,axis=1)
    print(con_mat)
    acc = np.diag(con_mat) / total
    acc1 = np.diag(con_mat1) / total
    acc2 = np.diag(con_mat2) / total
    acc3 = np.diag(con_mat3) / total
    
    overall_acc = np.mean(acc)
    overall_acc1 = np.mean(acc1)
    overall_acc2 = np.mean(acc2)
    overall_acc3 = np.mean(acc3)

    print("average Accuracy:")
    for ii, obj in enumerate(objects):
        print("%30s : full: %.3f, fft: %.3f, mrf: %.3f, phase: %.3f " % (obj, acc[ii], acc1[ii], acc2[ii], acc3[ii]))

    print("%30s : full: %.3f, fft: %.3f, mrf: %.3f, phase: %.3f " % ("OVERALL ACC", overall_acc,
                                                                    overall_acc1, overall_acc2, overall_acc3))

    all_aps += [ap]
    all_aps1 += [ap1]
    all_aps2 += [ap2]
    all_aps3 += [ap3]
    
    all_accs += [acc]
    all_accs1 += [acc1]
    all_accs2 += [acc2]
    all_accs3 += [acc3]
    
all_aps = np.asarray(all_aps)
all_aps1 = np.asarray(all_aps1)
all_aps2 = np.asarray(all_aps2)
all_aps3 = np.asarray(all_aps3)

stds = np.std(all_aps, axis=0)
stds1 = np.std(all_aps1, axis=0)
stds2 = np.std(all_aps2, axis=0)
stds3 = np.std(all_aps3, axis=0)

axis1_all_aps = np.mean(all_aps, axis=1)
axis1_all_aps1 = np.mean(all_aps1, axis=1)
axis1_all_aps2 = np.mean(all_aps2, axis=1)
axis1_all_aps3 = np.mean(all_aps3, axis=1)

axis0_all_aps = np.mean(all_aps, axis=0)
axis0_all_aps1 = np.mean(all_aps1, axis=0)
axis0_all_aps2 = np.mean(all_aps2, axis=0)
axis0_all_aps3 = np.mean(all_aps3, axis=0)

print("\n\n Final Average Precision:")
for ii, obj in enumerate(objects):
        print("%30s : full: %7.3f/%6.3f, fft: %7.3f/%6.3f, mrf: %7.3f/%6.3f, phase: %7.3f/%6.3f" % (obj, axis0_all_aps[ii]*100, stds[ii]*100,
                                                                                                            axis0_all_aps1[ii]*100, stds1[ii]*100,
                                                                                                            axis0_all_aps2[ii]*100, stds2[ii]*100,
                                                                                                            axis0_all_aps3[ii]*100, stds3[ii]*100))
        
print("%30s : full: %7.3f/%6.3f, fft: %7.3f/%6.3f, mrf: %7.3f/%6.3f, phase: %7.3f/%6.3f" % ("Mean all_ap", np.mean(axis1_all_aps)*100,
                                                                                                    np.std(axis1_all_aps)*100,                          
                                                                                                    np.mean(axis1_all_aps1)*100, np.std(axis1_all_aps1)*100,
                                                                                                    np.mean(axis1_all_aps2)*100, np.std(axis1_all_aps2)*100,
                                                                                                    np.mean(axis1_all_aps3)*100, np.std(axis1_all_aps3)*100,
                                                                                                    ))

all_accs = np.asarray(all_accs)
all_accs1 = np.asarray(all_accs1)
all_accs2 = np.asarray(all_accs2)
all_accs3 = np.asarray(all_accs3) 

stds = np.std(all_accs, axis=0)
stds1 = np.std(all_accs1, axis=0)
stds2 = np.std(all_accs2, axis=0)
stds3 = np.std(all_accs3, axis=0)

axis1_all_accs = np.mean(all_accs, axis=1)
axis1_all_accs1 = np.mean(all_accs1, axis=1)
axis1_all_accs2 = np.mean(all_accs2, axis=1)
axis1_all_accs3 = np.mean(all_accs3, axis=1)


axis0_all_accs = np.mean(all_accs, axis=0)
axis0_all_accs1 = np.mean(all_accs1, axis=0)
axis0_all_accs2 = np.mean(all_accs2, axis=0)
axis0_all_accs3 = np.mean(all_accs3, axis=0)

print(axis1_all_accs)
print(axis0_all_accs)


print("\n\n Final Average Accuracy:")
for ii, obj in enumerate(objects):
        print("%30s : full: %7.3f/%6.3f, fft: %7.3f/%6.3f, mrf: %7.3f/%6.3f, phase: %7.3f/%6.3f" % (obj, axis0_all_accs[ii]*100, stds[ii]*100,
                                                                                                            axis0_all_accs1[ii]*100, stds1[ii]*100,
                                                                                                            axis0_all_accs2[ii]*100, stds2[ii]*100,
                                                                                                            axis0_all_accs3[ii]*100, stds3[ii]*100))
        
print("%30s : full: %7.3f/%6.3f, fft: %7.3f/%6.3f, mrf: %7.3f/%6.3f, phase: %7.3f/%6.3f" % ("Mean all_acc", np.mean(axis1_all_accs)*100,
                                                                                                    np.std(axis1_all_accs)*100,                          
                                                                                                    np.mean(axis1_all_accs1)*100, np.std(axis1_all_accs1)*100,
                                                                                                    np.mean(axis1_all_accs2)*100, np.std(axis1_all_accs2)*100,
                                                                                                    np.mean(axis1_all_accs3)*100, np.std(axis1_all_accs3)*100,
                                                                                                    ))



