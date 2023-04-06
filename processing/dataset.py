# from re import M
import torch
import numpy as np
import torch.utils.data as data_utils
import glob
import scipy
import scipy.signal as sig
import os
import nn.utility as utility
import processing.processing as processing
import random
from scipy.signal import hilbert
import scipy.io as sio

def find_speaker_idx(X):
    X_fft = np.squeeze(X['X_power'].T)
    X_fft /= max(X_fft)
    peaks, _ = scipy.signal.find_peaks(X_fft[5:40],height=0.1)
    peaks += 5
    stds = []
    for p in peaks:
        ph_std = np.std(processing.butter_highpass_filter(X['X_phase_all'][:,p], 15, 250)) 
        stds.append(ph_std)
    max_std = max(stds)
    idx = peaks[stds.index(max_std)]
    return idx

class MatDataset(data_utils.Dataset):
    """This class is used to create the dataset for classification training."""
    def __init__(self, datapath, dataroot, input_len=250, normalize=False, cutoff=10, feature_names="fft", objects_of_interest=["wine-glass", "foam-roller"]):
        """
        * datapath: 
        where the data is located. It is a list of paths [path1, path2, ...].
        * input_len: 
        the length of the FFT after taking the first half (so default is FFT of length 500 which is then halved to 250.
        * normalize: 
        whether or not to normalize the final frequency spectrum.
        * cutoff:
        value to cutoff after taking the FFT of the phase of the bin where the object is.
        * feature_name:
        feature to load: stft, fft, welch, etc..
        """
        
        super().__init__()
        self.datapath = datapath
        self.dataroot = dataroot
        self.raw_datapath = os.path.dirname(os.path.realpath('data-capture/prod-cons-capture.py'))
        self.input_len = input_len
        self.normalize = normalize
        self.range_res = 0.056266129623825
        self.M = 2
        self.theta_s, self.theta_e = 20*(np.pi/180), 160*(np.pi/180)
        self.theta_rad_lim = [self.theta_s,self.theta_e]
        self.d_theta = 1/180*np.pi
        self.theta = np.arange(self.theta_rad_lim[0],self.theta_rad_lim[1],self.d_theta)
        self.feature_names = ["X_" + feature_name for feature_name in feature_names]
        self.filepaths = None
        self.objects = objects_of_interest
        self.cutoff = cutoff
        self.bad_ones = []
        self.is_metal = []
        self.is_aluminum = []
        self.is_flat = []
        self.is_cardboard = []
        self.is_ceramic = []
        self.is_foam = []
        self.is_glass = []
        self.is_wood = []
        self.is_plastic = []
        self.metals = ["aluminum", "black-usrp", "brass", "cast-iron", "copper", "febreeze", "metal-box", "metal-pot", "steel", "water-bottle", "white-usrp", "usrp", "other"]
        self.aluminum = ["aluminum"]
        self.flat = ["aluminum", "black-usrp", "brass", "cardboard", "ceramic", "copper", "glass", "hardwood", "metal-box", "plastic", "plastic-box", "steel", "white-usrp", "wood", "usrp"]
        self.plastic = ['clorox-wipes', 'plastic-bowl', 'plastic-food-container', 'plastic-box', 'trash-bin', 'plastic', 'large-plastic-bowl']
        self.glass = ['glass', 'wine-glass', 'glass-jar', 'glass-food-container', 'candle-jar', 'chicago-candle']
        self.foam = ['foam', 'foam-head', 'foam-roller']
        self.wood = ['wood', 'hardwood']
        self.cardboard = ['cardboard']
        self.ceramic = ['ceramic', 'cup', 'ceramic-bowl']
        self.material = ["cardboard", "ceramic", "foam", "glass", "wood", "metals", "plastic"]
        self.material_idx = []
        self.range_bins = {}
        self.objects_of_interest = objects_of_interest
        self.saved_features = ['X_phase_all', 'X_power', "X_damp", "X_idx",  "X_raw"]
        self.initialize()
        
    def initialize(self):
        """finds all the matfiles inside the datapath list provided.""" 
        self.filepaths = []
        
        for this_path in self.datapath:
            # create sub-directory for date we are loading
            utility.create_dir(os.path.join(self.dataroot,this_path.split("/")[-1]))
            self.filepaths += self.find_matfiles(this_path)
        random.shuffle(self.filepaths)
    
    def find_matfiles(self, somepath):
        """find all the matfiles inside a single given path."""
        return glob.glob(os.path.join(somepath, "*.mat"))
    
    def preprocess(self, idx, feature_names):
        """returns the processed input and groundtruth given the raw input.
        X: the input signal (1-D frequency spectrum)
        Y: object type
        name: name of the file to keep track of where each sample came from."""
        data_dict, exp_date, processed_flag = self.load_file(idx)
        ###
        name = self.filepaths[idx]
        filepath = self.idx_to_file(idx)
        
        date = filepath.split('/')[-2]
        exp_name = filepath.split('/')[-1]
        if(data_dict['exp_object'][0] not in self.objects_of_interest):
            if((data_dict['exp_object'][0] == 'white-usrp' or data_dict['exp_object'][0] == 'black-usrp') and 'usrp' in self.objects_of_interest):
                a = 1
            else:
                return None, None, name
        if(exp_name == '0.mat'):
            return None, None, name
        try:
            if(not processed_flag):
                save_idx_flag = False
                if (os.path.exists(os.path.join(self.dataroot, date,exp_name))):
                    file = utility.load_processed_data(date, exp_name, self.dataroot)
                    save_idx_flag = True
                    bin_idx = file['data']['X_idx'][0][0][0][0]
                X_dict = processing.process_raw_data(data_dict['result'], self.normalize, self.input_len, self.cutoff, self.range_res, self.M, 2, save_idx_flag, bin_idx, [5,30], True)
                Y = self.process_object(data_dict['exp_object'][0])
                exp_num = data_dict['exp_num']
                X_dict["exp_num"] = exp_num[0][0]
            else:
                X_dict = dict(zip(self.saved_features, [data_dict['data'][feature_name][0][0] for feature_name in self.saved_features]))
                Y = self.process_object(data_dict['exp_object'][0])
                bin_idx = X_dict['X_idx'][0][0]
                speaker_idx = find_speaker_idx(X_dict)
                
            num_nonoutliers, num_outliers, phase_cleaned1 = processing.clean_data(X_dict, 2)
            num_nonoutliers, num_outliers, phase_cleaned0 = processing.clean_data(X_dict, 1)
            num_nonoutliers, num_outliers, phase_cleaned2 = processing.clean_data(X_dict, 3)
            phase_cleaned = np.concatenate((phase_cleaned0[:,np.newaxis],phase_cleaned1[:,np.newaxis],phase_cleaned2[:,np.newaxis]),axis=1)

            if('X_fft' in feature_names):
                    X_dict['X_fft'] = self.process_sliding_windowed_fft(X_dict['X_phase_all'], X_dict['X_power'],bin_idx)
                    
            if('X_bf_soft' in feature_names):
                X_dict['X_bf_soft'] = self.process_fft_spec_bf(phase_cleaned)
            
            if('X_bf_soft_3' in feature_names):
                    X_dict['X_bf_soft_3'] = self.process_fft_spec_bf_3(phase_cleaned) 
                    
            if('X_damp' in feature_names):
                X_dict['X_damp'] = self.process_damp(X_dict['X_damp'], 0)
                    
            if('X_mrf_squared' in feature_names):
                X_mrf_squared, _, _ = processing.extract_rf_features(X_dict['X_power'], self.range_res)
                X_dict['X_mrf_squared'] = self.process_power(X_mrf_squared, bin_idx, speaker_idx)
                
            X = [X_dict[feature_name] for feature_name in feature_names]
            return X, Y, name
        except:
            return None, None, name
   
    def process_power(self, x, idx, speaker_idx):
        x = np.squeeze(x)
        start_idx = max(speaker_idx - 15,1)
        X = x[start_idx:start_idx + 25] 
        X = np.reshape(X, (1, X.shape[-1]))
        return X    
    
    def process_wasa(self, x, idx):
        x  = np.mean(x, axis=(0,1), keepdims=False)
        x = np.squeeze(x) # 4, 64
        x_t = np.abs(scipy.fft.fft(x, axis=-1))
        X_t = x_t[:,idx-3:idx + 4] 
        X = np.zeros([4,13])
        for i in np.arange(X.shape[0]):
            X_interp = scipy.interpolate.interp1d(np.arange(0,2*X_t.shape[1],2), np.squeeze(X_t[i,:]), kind='quadratic')
            X[i,:] = X_interp(np.arange(0,13,1))
        X = np.reshape(X, (1, 52))
        return X
    
    def process_fft_spec_bf(self, x):
        x = x.T # now is # range bins x # samples
        len_sig = 650
        num_samples = 650 + 1
        fs = 250
        cutoff = int(15 / 250. * len_sig)
        X = np.zeros((int(len_sig//2 - cutoff)))
        for i in range(3):
            freqs, t, fft_data = scipy.signal.stft(x[1,150:150+num_samples],window=scipy.signal.windows.hann(len_sig),fs=fs,nperseg=len_sig,noverlap=len_sig-50, nfft=len_sig, return_onesided=False)
            fft_data = abs(fft_data.T[:,cutoff:len_sig//2])**2
            u = np.mean(fft_data, 1)
            s = np.std(fft_data, 1)
            fft_data = (fft_data.T - u) / s
            X += np.squeeze(np.sum(fft_data,1))
        X = np.reshape(X, (1, X.shape[-1]))
        return X    
    
    def process_fft_spec_bf_3(self, x):
        x = x.T # now is # range bins x # samples
        len_sig = 650
        fs = 250
        cutoff = int(15 / 250. * len_sig)
        X = np.zeros((3,int(len_sig//2 - cutoff)))
        for i in range(3):
            freqs, t, fft_data = scipy.signal.stft(x[i,150:1650],window=scipy.signal.windows.hann(len_sig),fs=fs,nperseg=len_sig,noverlap=len_sig-50, nfft=len_sig, return_onesided=False)
            fft_data = abs(fft_data.T[:,cutoff:len_sig//2])**2
            u = np.mean(fft_data, 1)
            s = np.std(fft_data, 1)
            if(0 in s):
                print("here")
            fft_data = (fft_data.T - u) / s
            X[i,:] = np.squeeze(np.sum(fft_data,1))
        # X = np.reshape(X, (1, X.shape[-1]))
        return X
    
    def process_stft_bf(self, x):
        
        # x = x.T # now is # range bins x # samples
        len_sig = 650
        fs = 250
        cutoff = int(15 / 250. * len_sig)
        X = np.zeros((3,int(len_sig//2 - cutoff)))
        freqs, t, fft_data = scipy.signal.stft(x[150:1650],window=scipy.signal.windows.hann(len_sig),fs=fs,nperseg=len_sig,noverlap=len_sig-50, nfft=len_sig, return_onesided=False)
        fft_data = abs(fft_data.T[:,cutoff:len_sig//2])**2
        u = np.mean(fft_data, 1)
        s = np.std(fft_data, 1)
        fft_data = (fft_data.T - u) / s
        X = np.squeeze(fft_data.T)
        # print(X.shape)
        # X = np.reshape(X, (1, X.shape[-1]))
        return X
    
    def process_sliding_windowed_fft(self, x, power, ranges):
        # idx = np.argmax(power[0,ranges[0]:ranges[1]]) + ranges[0]
        idx = ranges
        len_sig = 650
        overlap = len_sig//4
        cutoff = int(15 / 250. * len_sig)
        offset = 150 - 1
        X_out = np.zeros((5,int(len_sig//2 - cutoff)))
        for i in range(X_out.shape[0]):
            fft_data = abs(scipy.fft.fft(x[i*overlap + offset:i*overlap + len_sig + offset, idx],len_sig))
            X_out[i,:] = fft_data[cutoff:len_sig//2]
        return X_out

    def process_damp(self, x, i):
        y = np.zeros(shape=(8, 125))
        for ii in range(y.shape[0]):
            idx = np.argmax(x[i,ii*250:ii*250+250])
            y[ii, :] = x[i,ii*250 + idx:ii*250+idx+125]
        # return y
        analytic_signal = hilbert(y)
        analytic_signal = np.abs(analytic_signal)
        return analytic_signal

    
    def process_object(self, this_object):
        """process the object and add it to the list of all objects if not included already."""
        if(this_object == 'white-usrp' or this_object == 'black-usrp'):
            this_object = 'usrp'
        if('aluminium' in this_object):
            this_object = 'aluminum'
        elif('aluminum' in this_object):
            this_object = 'aluminum'
        if('copper' in this_object):
            this_object = 'copper'
        if('brass' in this_object):
            this_object = 'brass'
        if('steel' in this_object):
            this_object = 'steel'
        return this_object
            
    def __getitem__(self, idx):
        """return one sample given an index."""
        X, Y, name = self.preprocess(idx, self.feature_names)
        obj = Y
        #metals
        if obj in self.metals:
            self.is_metal += [1]
        else:
            self.is_metal += [0]
        #flats
        if obj in self.flat:
            self.is_flat += [1]
        else:
            self.is_flat += [0]
        if obj in self.metals:
            self.material_idx += [self.material.index("metals")]
        elif obj in self.plastic:
            self.material_idx += [self.material.index("plastic")]
        elif obj in self.wood:
            self.material_idx += [self.material.index("wood")]
        elif obj in self.glass:
            self.material_idx += [self.material.index("glass")]
        elif obj in self.foam:
            self.material_idx += [self.material.index("foam")]
        elif obj in self.ceramic:
            self.material_idx += [self.material.index("ceramic")]
        elif obj in self.cardboard:
            self.material_idx += [self.material.index("cardboard")]
        else:
            self.material_idx += [-1]
        return {'X': X, 'Y': Y, 'name': name, 'metal': self.is_metal[idx], 'flat': self.is_flat[idx], 
                 'material': self.material_idx[idx]}
            
    def load_file(self, idx):
        """load a file given the index"""
        filepath = self.idx_to_file(idx)
        date = filepath.split('/')[-2]
        exp_name = filepath.split('/')[-1]
        processed_flag = False
        # check if we have already processed data
        if (os.path.exists(os.path.join(self.dataroot, date,exp_name))):
            file = utility.load_processed_data(date, exp_name,root_path=self.dataroot)
            processed_flag = True
        else:
            print(filepath)
            # file = sio.loadmat(filepath)
        return file, date, processed_flag
        
    def idx_to_file(self, idx):
        """return the filepath given the index"""
        return self.filepaths[idx]
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.filepaths)


class FastDataset(data_utils.Dataset):
    """This version uses the MatDataset and returns the already loaded data."""
    def __init__(self, X, Y, names, objects):   
        """ 
        * n_obj: 
            total number of objects (e.g. 4 for ['aluminium', 'copper', 'brass', 'steel']).
        * named_objects:
            objects in the string format (['aluminium', 'aluminium', 'steel', ...])
        * X
            input to the network
        * Y
            objects in the numerical format ([0, 0, 3, ...])
        * names
            names of the files corresponding to each (X, Y) pair.
        * objects
            list of unique objects (['aluminium', 'copper', 'brass', 'steel'])
        """
        super().__init__()
        self.n_obj = len(objects)
        self.named_objects = Y
        self.X = X
        self.Y = []
        # self.base_idx = []
        self.names = names
        self.objects = objects
        
        self.initiate()
        
    def initiate(self):
        """create the numerical equivalent of the objects in the string format."""
        for obj in self.named_objects:
            self.Y += [self.objects.index(obj)] 
            
    def to_categorical(self, y):
        """ 1-hot encodes a tensor """
        return np.eye(self.n_obj, dtype='uint8')[y]


    def __getitem__(self, idx):
        """ return the input and output and the name given the index."""
        return {'X': self.X[idx], 'Y': self.Y[idx], 'name': self.names[idx],
                'obj': self.named_objects[idx], }
    
    def __str__(self):
        return str(self.Y[0]) +  str(self.Y[1]) + str(self.Y[2])
    
    def __len__(self):
        return len(self.Y)
    
def createDataset(dataroot="/media/synrg-sc1/HardDisk2-8TB/Mat_Sensing/backup/Box",
                  dates=None, input_len=250, normalize=False, val_samples=0.3,train_for='Y',
                  batch_size=4, cutoff=10, sample_limit=10000000, feature_names=['fft', 'mrf_squared'], lim=10000000, objects_of_interest=["aluminum", "brass", "copper", "steel"]):
    """create the dataset given the dataroot, dates, input_len and normalize (see __initiate__ in matDataset)"""
    if dates is None:
        dates = ["jan-25-2", "jan-28", "jan-28-1"]
        
    datapaths = [os.path.join(dataroot, date) for date in dates]
    
    all_data = []
    all_objects = []
    all_names = []
    if(isinstance(feature_names, str)):
        feature_names = [feature_names]
    A = MatDataset(datapath=datapaths, dataroot=dataroot, input_len=input_len, normalize=normalize, cutoff=cutoff, feature_names=feature_names, objects_of_interest=objects_of_interest)
    
    for i, sample in enumerate(A):
        X = sample['X']
        name = sample['name']
        if X is not None:
            Y = sample[train_for]
            if(all_objects.count(Y) > lim):
                continue
            all_data += [X]
            all_objects += [Y]
            all_names += [name]
            if i >= sample_limit:
                print("hit sample limit of ", i)
                break
            print(f"processed sample {i}...", end='\r')
        else:
            print(f"skipped sample {i}!!", end='\r')
    print("\n")
        
    if train_for == 'material':
        objects = [0,1,2,3,4,5,6]
    elif train_for == 'metal':
        objects = [0,1]
    else:
        objects = A.objects
        
        
    for obj in objects:
        print(str(obj) + ": ", (all_objects.count(obj)))
        
    print("Total Experiments:", (len(all_objects)))
    fast_dataset = FastDataset(all_data, all_objects, all_names, objects)
    total_length = len(fast_dataset)

    if val_samples > 0:
        val_length = int(total_length * val_samples) - 1
        test_length = val_length
        train_length = total_length - val_length - test_length
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(fast_dataset, [train_length, val_length, test_length])
        train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, val_loader, test_dataset
    else:
        data_loader = data_utils.DataLoader(fast_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return data_loader


if __name__ == "__main__":
    """sample code to run and debug"""
    dataroot = "/media/synrg-sc1/HardDisk2-8TB/Mat_Sensing/backup/Box/"
    all_data = []
    all_objects = []
    all_names = []
            
    dates = """mar-8, mar-8-2, mar-15, mar-15-3, mar-15-6, mar-18-3, mar-3-2, mar-3-5, mar-3-7, feb-10-2, feb-10-3, feb-10-4, feb-11, feb-11-1, feb-11-2, feb-22-1, feb-23, feb-23-1, mar-15-5, 
             mar-16-1, mar-16-2, mar-16-4, mar-18-6, mar-2-2, mar-2-3, mar-22, mar-22-1, mar-22-2, mar-22-6, mar-23-1, mar-23-2,  mar-23-4, mar-23-5,  mar-3-6, mar-3-9, mar-4, feb-23-2, feb-25, 
             mar-15-4, mar-16-5, mar-16-6, mar-16-7, mar-16-8, mar-16-9, mar-17-2, mar-17-3, mar-17-5, mar-17-6, mar-18-2, mar-22-3, mar-23, mar-23-3, apr-26, apr-29,  may-3, may-9, may-11, may-11-1,
             may-11-2, may-11-3, june-11,june-11-1,june-11-2,june-11-3,june-12-1,june-12-2,june-14-1,may-19-metals2,may-20-1,may-20-2,may-20-3,may-20-4,may-23-1,may-23-2,may-23-3,may-23-4,may-23-5,
             may-24-1,may-24-2,may-24-3,may-25-1,may-25-2,may-25-3,may-26,may-26-1,may-26-2,may-26-3,may-26-4,may-27-1,may-27-6,may-27-7,may-30-1,may-30-2,may-31-2,may-31-3,may-31-4,june-1,june-1-2,
             june-1-3,june-1-4,june-2-3,june-2-6,june-2-7,june-3-1,june-3-2,june-3-3,june-3-4,june-5-1,june-7-4,june-7-5,june-8-1,june-8-2,june-9-1,june-9-2,june-9-3,june-10-1,june-10-2,june-10-3,
             june-10-4,june-13-power1,june-13-2,june-13-3,may-14-1,may-14-2,may-15,may-15-1,may-15-2,may-15-3,may-15-4,may-15-5,may-16,may-16-2,may-16-3,may-18,
             mar-17, mar-17-1, june-1-1,june-2-4,june-6-2,june-6-3,june-6-4,june-7-1,june-7-2,june-7-3, mar-4-1,mar-7-2, mar-7-3,may-27-3,may-27-4,may-27-5, mar-22-2, mar-23-2, may-30-3, may-30-4, 
            mar-17-7,mar-17-4, mar-18, mar-18-1, mar-18-4, mar-18-7, mar-2-4, mar-22-4, mar-22-5,  may-10, may-11-4, may-11-5, may-13, may-14, may-16-1, may-27-2, may-31-1,june-13-4,jun-13-5,june-13-6,june-14,june-14-2,june-14-3,june-14-4"""
    dates = utility.parse_dates(dates)
    print(dates)
    objects = 'aluminum, usrp, brass, candle-jar, cardboard, ceramic, ceramic-bowl, clorox-wipes, copper, cup, febreeze, foam, foam-head, foam-roller, glass, glass-food-container, glass-jar, hardwood, large-plastic-bowl, metal-box, metal-pot, plastic, plastic-box, plastic-food-container, steel, trash-bin, water-bottle, wine-glass, wood'
    objects = utility.parse_objects(objects)
    datapaths = [os.path.join(dataroot, date) for date in dates]
    A = createDataset(dataroot=dataroot, dates=dates,feature_names = ['mrf_squared'], objects_of_interest=objects)
    print("\n", A)
    
    