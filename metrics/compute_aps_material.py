import os
import scipy.io as sio
import numpy as np
import utility
import argparse

dir_path = os.path.dirname(os.path.realpath('nn/train.py'))
path_to_use = os.path.join(dir_path, 'results')
opt = utility.view_metrics_options(parser=argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter))
# experiment name, dataset path, and logging
exp_name = utility.new_exp(opt.exp_name)

all_aps, all_aps1, all_aps2, all_aps3, all_accs, all_accs1, all_accs2, all_accs3, = [], [], [], [], [], [], [], []


for exp_rep in range(10):
    if(exp_rep == 2 or exp_rep == 7 or exp_rep == 5):
        continue
    metadata_path = os.path.join(dir_path,"metadata", exp_name, str(exp_rep), ".mat")
    opt = sio.loadmat(metadata_path)['opt']
    # print(opt)
    objects = utility.parse_objects(str(opt[0][0][8][0]))
    outputs = list(range(0, len(objects)))
    objects_nonmetals =  ['candle-jar', 'glass-food-container', 'plastic-food-container', 'glass-jar', 'wine-glass', 'cup', 'clorox-wipes', 'cardboard', 'hardwood', 'foam', 'glass', 'ceramic', 'ceramic-bowl', 'foam-head', 'foam-roller','large-plastic-bowl', 'plastic', 'plastic-bowl', 'plastic-box', 'wood']
    objects_metals =  ['aluminum', 'brass', 'copper', 'steel', 'metal-box', 'usrp', 'white-usrp', 'black-usrp', 'metal-pot', 'water-bottle', 'cast-iron', 'febreeze']
    objects_glass =  ['chicago-candle', '   candle-jar', 'glass-food-container', 'glass-jar', 'wine-glass' , 'glass']
    objects_plastic = ['clorox-wipes', 'plastic', 'plastic-bowl', 'plastic-food-container', 'plastic-box', 'trash-bin', 'large-plastic-bowl']
    objects_foam = ['foam', 'foam-head', 'foam-roller']
    objects_ceramic = ['cup', 'ceramic', 'ceramic-bowl', ]
    objects_wood = ['hardwood', 'wood']
    objects_cardboard = ['cardboard']
    objects_tot = objects
    tot = [objects.index(i) for i in objects_tot]
    objects =  ['metal', 'cardboard', 'glass', 'ceramic', 'plastic', 'foam', 'wood']
    n_obj = len(objects)
    outputs = outputs

    con_mat = np.zeros(shape=(n_obj, n_obj))
    con_mat1 = np.zeros(shape=(n_obj, n_obj))
    con_mat2 = np.zeros(shape=(n_obj, n_obj))
    con_mat3 = np.zeros(shape=(n_obj, n_obj))
    true_labels = []
    scores = []
    scores1 = []
    scores2 = []
    scores3 = []
    print(con_mat[0, 0])
    test_date = []
    for filename in range(50000):
        filepath = os.path.join(path_to_use, exp_name, str(exp_rep), str(filename))
        if(os.path.isfile(filepath + '.mat')):
            A = sio.loadmat(filepath)
            x1 = A['x1']
            output = A['output']
            output1 = A['output']
            output2 = A['output']
            output3 = A['output']
            gt = A['gt']
            name = A['name']
            batch_size = x1.shape[0]
            for batch in range(batch_size):
                y = np.argmax(output[batch,:])
                y1 = np.argmax(output1[batch,:])
                y2 = np.argmax(output2[batch,:])
                y3 = np.argmax(output3[batch,:])
                reind_y = tot.index(int(y))
                reind_y1 = tot.index(int(y1))
                reind_y2 = tot.index(int(y2))
                reind_y3 = tot.index(int(y3))
                if(objects_tot[int(gt[0][batch])] in objects_metals): 
                    reind_gt = objects.index('metal')
                elif(objects_tot[int(gt[0][batch])] in objects_cardboard):
                    reind_gt = objects.index('cardboard')
                elif(objects_tot[int(gt[0][batch])] in objects_wood):
                    reind_gt = objects.index('wood')
                elif(objects_tot[int(gt[0][batch])] in objects_glass):
                    reind_gt = objects.index('glass')
                elif(objects_tot[int(gt[0][batch])] in objects_ceramic):
                    reind_gt = objects.index('ceramic')
                elif(objects_tot[int(gt[0][batch])] in objects_plastic):
                    reind_gt = objects.index('plastic')
                elif(objects_tot[int(gt[0][batch])] in objects_foam):
                    reind_gt = objects.index('foam')
                    
                if(objects_tot[int(y)] in objects_metals):
                    reind_y = objects.index('metal')
                elif(objects_tot[int(y)] in objects_cardboard):
                    reind_y = objects.index('cardboard')
                elif(objects_tot[int(y)] in objects_wood):
                    reind_y = objects.index('wood')
                elif(objects_tot[int(y)] in objects_glass):
                    reind_y =  objects.index('glass')
                elif(objects_tot[int(y)] in objects_ceramic):
                    reind_y = objects.index('ceramic')
                elif(objects_tot[int(y)] in objects_plastic):
                    reind_y = objects.index('plastic')
                elif(objects_tot[int(y)] in objects_foam):
                    reind_y = objects.index('foam')
                    
                if(objects_tot[int(y1)] in objects_metals):
                    reind_y1 = objects.index('metal')
                elif(objects_tot[int(y1)] in objects_cardboard):
                    reind_y1 = objects.index('cardboard')
                elif(objects_tot[int(y1)] in objects_wood):
                    reind_y1 = objects.index('wood')
                elif(objects_tot[int(y1)] in objects_glass):
                    reind_y1 =  objects.index('glass')
                elif(objects_tot[int(y1)] in objects_ceramic):
                    reind_y1 = objects.index('ceramic')
                elif(objects_tot[int(y1)] in objects_plastic):
                    reind_y1 = objects.index('plastic')
                elif(objects_tot[int(y1)] in objects_foam):
                    reind_y1 = objects.index('foam')
                    
                    
                if(objects_tot[int(y2)] in objects_metals):
                    reind_y2 = objects.index('metal')
                elif(objects_tot[int(y2)] in objects_cardboard):
                    reind_y2 = objects.index('cardboard')
                elif(objects_tot[int(y2)] in objects_wood):
                    reind_y2 = objects.index('wood')
                elif(objects_tot[int(y2)] in objects_glass):
                    reind_y2 =  objects.index('glass')
                elif(objects_tot[int(y2)] in objects_ceramic):
                    reind_y2 = objects.index('ceramic')
                elif(objects_tot[int(y2)] in objects_plastic):
                    reind_y2 = objects.index('plastic')
                elif(objects_tot[int(y2)] in objects_foam):
                    reind_y2 = objects.index('foam')
                    
                    
                if(objects_tot[int(y3)] in objects_metals):
                    reind_y3 = objects.index('metal')
                elif(objects_tot[int(y3)] in objects_cardboard):
                    reind_y3 = objects.index('cardboard')
                elif(objects_tot[int(y3)] in objects_wood):
                    reind_y3 = objects.index('wood')
                elif(objects_tot[int(y3)] in objects_glass):
                    reind_y3 =  objects.index('glass')
                elif(objects_tot[int(y3)] in objects_ceramic):
                    reind_y3 = objects.index('ceramic')
                elif(objects_tot[int(y3)] in objects_plastic):
                    reind_y3 = objects.index('plastic')
                elif(objects_tot[int(y3)] in objects_foam):
                    reind_y3 = objects.index('foam')
                con_mat[reind_gt, reind_y] += 1
                con_mat1[reind_gt, reind_y1] += 1
                con_mat2[reind_gt, reind_y2] += 1
                con_mat3[reind_gt, reind_y3] += 1
                
                true_label = np.zeros(shape=(n_obj,))
                true_label[reind_gt] = 1
                true_labels += [list(true_label)]
    print(exp_rep)    
    total = np.sum(con_mat,axis=1)
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

    
    all_accs += [acc]
    all_accs1 += [acc1]
    all_accs2 += [acc2]
    all_accs3 += [acc3]
    
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

