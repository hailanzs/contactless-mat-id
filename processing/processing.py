from venv import create
import numpy as np
import scipy.signal as sig
import scipy

from statistics import stdev as std
import heapq

### start helper functions ###
    
def isNotOutlier(point, upper, lower):
    return (point < upper and point > lower)

def findNearestValue(data, before, current, after, threshAbove, threshBelow):
    before = before if before > -1 else 0
    after = after if after < len(data) else len(data) - 1

    while(True):
        if(after < len(data) and isNotOutlier(data[after],threshAbove,threshBelow)):
            return data[after]
        after += 1
        if(before >= 0 and isNotOutlier(data[before],threshAbove,threshBelow)):
            return data[before]
        before -= 1


def getOutlierLists(data, distancePos, distanceNeg):
    outlierList = []
    outlierList.extend(data[data > distancePos].tolist())
    outlierList.extend(data[data < distanceNeg].tolist())

    outlierListIndecies = [i for i, j in enumerate(data) if j in outlierList]

    return(outlierList, outlierListIndecies)


def filloutliers(data):
    stad = std(data)
    mean = np.mean(data)
    distancePos = 5*stad + mean
    distanceNeg = (-5*stad) + mean

    (outlierList, outlierListIndecies) = getOutlierLists(data, distancePos, distanceNeg)
    
    toReplace =[]

    for i in range(len(outlierList)):
        toReplace.append(findNearestValue(data, outlierListIndecies[i] - 1, outlierListIndecies[i], outlierListIndecies[i] + 1, distancePos, distanceNeg))

    for i in range(len(outlierListIndecies)):
        data[outlierListIndecies[i]] = toReplace[i]
        
    return data

def findOutliers(data, bin2):
    stad = std(bin2)
    mean = np.mean(bin2)
    distancePos = 5*stad + mean
    distanceNeg = (-5*stad) + mean

    (outlierList, outlierListIndecies) = getOutlierLists(bin2, distancePos, distanceNeg)
    if(len(outlierList) == 0):
        return data
    # small_outliers = 
    if(min(outlierListIndecies) < 1350 and min(outlierListIndecies) > 90):
        return None
    
    stad = std(data)
    mean = np.mean(data)
    distancePos = 3*stad + mean
    distanceNeg = (-3*stad) + mean
    
    for o in range(len(outlierListIndecies)):
        if(len(outlierListIndecies) < 5 or o + 5 < len(outlierListIndecies) and np.abs(outlierListIndecies[o] - outlierListIndecies[o+5]) > 500 and outlierListIndecies[o] > 90):
            # fill this single outlier and continue
            data[outlierListIndecies[o]] = findNearestValue(data, outlierListIndecies[o] - 1, outlierListIndecies[o], outlierListIndecies[o] + 1, distancePos, distanceNeg)
            continue
        elif(outlierListIndecies[o] > 150):
            # this means the point is important
            data = data[:outlierListIndecies[o]]
            break
    return data

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    high_s = min(high + 0.1, high + 4.000000000008001e-05)
    low_s = low - 0.1
    Ns, Wn = sig.buttord([low, high], [low_s, high_s], 3, 60)
    b, a = sig.butter(2, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cut, fs, order=2, axis_=0):
    nyq = 0.5 * fs
    low = cut / nyq
    b, a = sig.butter(N=order, Wn=low, btype="high")
    y = sig.lfilter(b, a, data,axis=axis_)
    return y

def find_peak_freqs(X, welch_cut_off):
    """ Finds peaks of welch signal -> input to self.extract_damp for damping feature"""
    peaks, _ = sig.find_peaks(X[welch_cut_off:])
    prominences = sig.peak_prominences(X[welch_cut_off:],peaks)
    max_vals = heapq.nlargest(3,prominences[0])
    max_peaks = [peaks[np.where(prominences[0]==max_val)[0][0]]  + welch_cut_off for max_val in max_vals]
    return max_peaks

def beamform(X,idx):
    """ X is of shape # frames x # Rx x # Tx x # bins 
    """
    ph_bf = 0
    beat_freq = scipy.fft.fft(X, axis=3)
    num_frms, tx, rx, N_range = beat_freq.shape

    idxs = np.arange(idx-2,idx+3)
    
    N_x_stp = rx
    N_z_stp = tx
    theta_s, theta_e = 70, 110
    theta_s *= (np.pi/180)
    theta_e *= (np.pi/180)
    theta_rad_lim = [theta_s,theta_e]
    d_theta = 2/180*np.pi
    phi_s, phi_e = 70, 110
    phi_s *= (np.pi/180)
    phi_e *= (np.pi/180)
    phi_rad_lim = [phi_s,phi_e]
    d_phi = 5/180*np.pi
    theta = np.arange(theta_rad_lim[0],theta_rad_lim[1],d_theta)
    N_theta = len(theta)
    phi = np.arange(phi_rad_lim[0],phi_rad_lim[1],d_phi)
    N_phi = len(phi)
    
    lm = 3e8/77e9
    sph_pwr = np.zeros((num_frms, N_theta, N_phi, 5), dtype=complex)
    sph_pwr_range = np.zeros((N_theta, N_phi, N_range), dtype=complex)
    
    x_idx = np.array([[0.,1.,2.,3.],[-2.,-1.,0.,1.]])
    z_idx = np.array([[0.,0.,0.,0.],[1.,1.,1.,1.]])
    s = lm / 2
    for kt in range(N_theta):
        for kp in range(N_phi):
       
            cos_theta = np.cos(theta[kt])
            sin_theta = np.sin(theta[kt])
            sin_phi = np.sin(phi[kp])
            
            sinp_cost = sin_phi * cos_theta
            sinp_sint = sin_phi * sin_theta
            
            # cos_theta = np.cos(theta[ka])
        
            Vec = np.exp(-1j*(2*np.pi*(s*z_idx*sinp_cost + s*x_idx*sinp_sint)/lm))
            VecRF = np.repeat(Vec[np.newaxis,:,:],num_frms,axis=0)
            VecRFR = np.repeat(VecRF[:,:,:,np.newaxis],N_range,axis=3)
            VecRFI = np.repeat(VecRF[:,:,:,np.newaxis],5,axis=3)
            sph_pwr[:,kt,kp,:] = np.squeeze(np.sum(np.multiply(beat_freq[:,:,:,idxs],VecRFI),axis=(1,2)))
            sph_pwr_range[kt,kp,:] = np.squeeze(np.sum(np.multiply(beat_freq,VecRFR),axis=(0,1,2)))
    
    pwr = np.squeeze(np.mean(abs(sph_pwr[:,:,:,2]),axis=0))**2
    max_loc = np.unravel_index(pwr.argmax(), pwr.shape)
    
    ph_bf = np.unwrap(np.angle(np.squeeze(sph_pwr[:,max_loc[0], max_loc[1],:])), axis=0)
    ph_bf = ph_bf - np.mean(ph_bf,axis=0)
    
    return sph_pwr[:,max_loc[0], max_loc[1],:], ph_bf, sph_pwr_range[max_loc[0], max_loc[1],:]



def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def GramSchmidt(X):
    Y = []
    for i in range(X.shape[0]):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            #print "i =", i, ", projection vector =", proj_vec
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
            #print "i =", i, ", temporary vector =", temp_vec
        Y.append(temp_vec)
    return Y

def compensate_range_bins(base_bins, obj_bins):
    base_bins = GramSchmidt(base_bins)
    y = obj_bins
    for ii in range(obj_bins.shape[1]):
        for jj in range(base_bins.shape[1]):
            y[:, ii] = y[:, ii] - np.dot[base_bins[:, jj], obj_bins[:, ii]] * base_bins[:, jj]
    return y
    

def isNotOutlier(point, upper, lower):
    return (point < upper and point > lower)

def findNearestValue(data, before, current, after, threshAbove, threshBelow):
    before = before if before > -1 else 0
    after = after if after < len(data) else len(data) - 1

    while(True):
        if(after < len(data) and isNotOutlier(data[after],threshAbove,threshBelow)):
            return data[after]
        after += 1
        if(before >= 0 and isNotOutlier(data[before],threshAbove,threshBelow)):
            return data[before]
        before -= 1
        if(after > len(data) and before < 0):
            return None

def getOutlierLists(data, distancePos, distanceNeg):
    outlierList = []
    outlierList.extend(data[data > distancePos].tolist())
    outlierList.extend(data[data < distanceNeg].tolist())

    outlierListIndecies = [i for i, j in enumerate(data) if j in outlierList]

    return(outlierList, outlierListIndecies)


def findOutliers(data, threshold):
    indices = []
    stad = std(data)
    # if(max(abs(data)) / np.median(data) > 150):
    if(stad > 0.009):
        median = np.median(data)
        MAD = np.median(np.absolute(data - np.median(data)))
        distancePos = threshold*MAD + median
        distanceNeg = (-threshold*MAD) + median

        (outlierList, indices) = getOutlierLists(data, distancePos, distanceNeg)
    return np.asarray(indices)
    
def softmax(x,t):
    return np.exp(x) / np.sum(np.exp(x),0)
    

def rolling_std(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def outliers_exist(x, threshold):

    std = np.std(x)
    if(std > threshold):
        return True
    return False

def filloutliers(data_2, data, threshold):
    data_2 = np.squeeze(data_2)
    data = np.squeeze(data)
    outlierListIndecies = []
    if(outliers_exist(data_2, 0.009)):
        median = np.median(data_2)
        MAD = np.median(np.absolute(data_2 - np.median(data_2)))
        distancePos = threshold*MAD + median
        distanceNeg = (-threshold*MAD) + median
    
        (outlierList, outlierListIndecies) = getOutlierLists(data_2, distancePos, distanceNeg)
        toReplace =[]

        for i in range(len(outlierList)):
            toReplace.append(findNearestValue(data_2, outlierListIndecies[i] - 1, outlierListIndecies[i], outlierListIndecies[i] + 1, distancePos, distanceNeg))

        for i in range(len(outlierListIndecies)):
            data_2[outlierListIndecies[i]] = toReplace[i]
            
    return len(outlierListIndecies), data_2

def clean_data(X, bin_idx):
    # phase = X['X_ph_bf']
    # take first 2500 samples (since that should include at least 9 pulses and have less outliers)
    # phase = phase[0:2500,:]
    # filter phase so low freq offsets dont matter
    # filt_phase_all = butter_highpass_filter(phase, 15, 250, axis_=0)  
    phase2 = X['X_phase_all']
    # take first 2500 samples (since that should include at least 9 pulses and have less outliers)
    phase2 = phase2[0:2500,:]
    # filter phase so low freq offsets dont matter
    filt_phase_all2 = butter_highpass_filter(phase2, 15, 250, axis_=0)       
    num_outliers, phase_cleaned = filloutliers(filt_phase_all2[:,1], filt_phase_all2[:,bin_idx], threshold=8)
    num_nonoutliers = phase2.shape[0] - num_outliers
    return num_nonoutliers, num_outliers, phase_cleaned
    
### end helper functions ###

def extract_rf_features(X, range_res):
    """multiplies range FFT of num_frames x 64 by distance  and distance squared and takes argmax to find maximum bin +- M
    returns: amplitudes of size num_bins that we care about"""
    X = np.abs(np.sum(X,axis=0))
    max_idxs = np.arange(0, 64) 
    mrf_features_squared  = X * (range_res*max_idxs)**2
    mrf_features = X * (range_res*max_idxs)
    power_features = X
    
    return mrf_features_squared, mrf_features, power_features

def real_fft(X, cutoff, input_len):
    """take fft of real signal + remove the first half + remove the beginning."""
    X = abs(scipy.fft.fft(X))
    return np.squeeze(X)  

def extract_welch(X, input_len):
    """take fft of real signal + remove the first half + remove the beginning."""
    freqs, X = sig.welch(X)  # length 129
    return freqs, np.squeeze(X) 

def extract_stft(X):
    """take fft of real signal + remove the first half + remove the beginning."""
    freqs, times, Zxx = sig.stft(X) # length 129 x 27
    return Zxx 
    

def extract_damp(X, X_freq, N, freqs, welch_cut_off):
    """Extract damping signals from the peak frequencies in the welch signal
    outputs # peaks x lenght of time domain signal size"""
    peaks = find_peak_freqs(X_freq, welch_cut_off)
    X_damp = np.zeros((len(peaks), len(X)))
    X_damp_raw = np.zeros((1, len(X)))
    for i in range(len(peaks)):
        # filter out around the peak
        X_damp[i,:] = (butter_bandpass_filter(X,min(freqs[peaks[i]]-N, 124),min(freqs[peaks[i]]+N, 124.99),250))  
    X_damp_raw = butter_highpass_filter(X, 15, 250)  
    
    return X_damp, X_damp_raw


def process_raw_data(X, normalize, input_len, cutoff, range_res, M, N, save_idx_flag, idx, range_bins, saved_bins):
    """pre-process raw mm-wave data to get the frequency spectrum."""
    # X_raw = X
    X_ = np.mean(X, axis=(1,2), keepdims=False)
    X_ = X_ - np.mean(X_, axis=-1, keepdims=True)
    X_rfft = scipy.fft.fft(X_, axis=-1)
    if(not save_idx_flag):
        if(saved_bins):
            idx = np.argmax(np.sum(np.abs(X_rfft), axis=0)[range_bins[0]-1:range_bins[1]]) + range_bins[0] - 1
            # print(f'THIS IS PROCESSING idx {idx}', idx, end='\r')
        else:
            return None
    # X_bf, X_ph_bf, X_sph_pwr_range = beamform(X, idx)
    X_mrf_squared, X_mrf, X_power = extract_rf_features(X_rfft, range_res)
    X_phase_all = np.unwrap(np.angle(X_rfft), axis=0)
    X_phase_all -= np.mean(X_phase_all, axis=0, keepdims=True)

    X_phase = np.unwrap(np.angle(X_rfft[:, idx]))
    X_phase -= np.mean(X_phase)
    
    # TODO: add beamforming here!
    X_fft = real_fft(X_phase, cutoff, input_len)
    X_stft = extract_stft(X_phase)
    freqs, X_welch = extract_welch(X_phase, input_len)
    X_damp, X_damp_raw = extract_damp(X_phase, X_welch, N, freqs*250, 18)
    X_raw = X_[:,idx-2:idx+3]
    if normalize: 
        X_fft = X_fft / np.max(X_fft)
        X_welch = X_welch / np.max(X_welch)
    return {'X_phase_all': X_phase_all, 'X_power': X_power, 
            "X_damp": X_damp, "X_idx": idx, "X_raw": X_raw}#, "X_sph_pwr": X_bf, "X_ph_bf": X_ph_bf}