from venv import create
import numpy as np
import scipy.signal as sig
import scipy

from statistics import stdev as std
import heapq

### start helper functions ###
    
def isNotOutlier(point, upper, lower):
    """Whether or not a point is within the given limits.
    - Inputs:
        - point: scalar number
        - upper: upper bound
        - lower: lower bound
    - Outputs:
        - [Boolean scalar] whether or not point is stricly between lower and upper."""
    return (point < upper and point > lower)

def findNearestValue(data, before, current, after, threshAbove, threshBelow):
    """
    Return non-outlier data sample (i.e. sample that falls between threshAbove and threshBelow).
    - Inputs:
        - data: iterable list/array in which we are going to search
        - before: we are looking at indices 'before' or smaller than before.
        - after: or, alternatively, indices 'after' or larger than after.
        - threshAbove: ignore instances where data is larger than this value.
        - threshBelow: ignore insstancs where data is smaller than this value.   
    """
    # Ensure 'before' index is not negative
    before = before if before > -1 else 0
    # Ensure 'after' index is not greater than the length of the data
    after = after if after < len(data) else len(data) - 1

    while(True):
        # Check if the value at 'after' index is not an outlier
        if(after < len(data) and isNotOutlier(data[after], threshAbove, threshBelow)):
            return data[after]
        # Move to the next index after 'after'
        after += 1
        # Check if the value at 'before' index is not an outlier
        if(before >= 0 and isNotOutlier(data[before], threshAbove, threshBelow)):
            return data[before]
        # Move to the previous index before 'before'
        before -= 1


def getOutlierLists(data, distancePos, distanceNeg):
    """
    Find the values along with indices of outliers in a given input array.
    - Inputs:
        - data: input array for which we are going to find the outlier indices.
        - distancePos: the upper limit, after which values count as outliers.
        - distanceNeg: the lower limit, after which values count as outliers.
    - Outputs:
        - outlierList: The list containing all outlier values.
        - outlierListIndecies: The list containing corresponding outlier indices.

    """
    # First, we find all the indices that are outliers, given the upper and lower limit.
    # any data point that is not within the distanceNeg to distancePos limit, is an outlier.
    outlierList = []
    outlierList.extend(data[data > distancePos].tolist())
    outlierList.extend(data[data < distanceNeg].tolist())

    # For all the data point values, we find the corresponding indices in data.
    outlierListIndecies = [i for i, j in enumerate(data) if j in outlierList]
    #
    return(outlierList, outlierListIndecies)


def filloutliers(data):
    """
    Identifies and fills the outliers of a given input array.
    - Inputs:
        - data: input array for which we want to perform outlier filling.
    - Outputs: 
        - data: output array with filled outliers.
    """
    
    # Calculate the standard deviation and mean of the input data
    stad = std(data)
    mean = np.mean(data)
    
    # Define the thresholds for outlier detection
    distancePos = 5 * stad + mean  # Upper threshold
    distanceNeg = (-5 * stad) + mean  # Lower threshold

    # Get the list of outliers and their corresponding indices
    (outlierList, outlierListIndecies) = getOutlierLists(data, distancePos, distanceNeg)
    
    toReplace =[]

    # Find the nearest value to replace each outlier
    for i in range(len(outlierList)):
        toReplace.append(findNearestValue(data, outlierListIndecies[i] - 1, outlierListIndecies[i], outlierListIndecies[i] + 1, distancePos, distanceNeg))

    # Replace the outliers with the computed values
    for i in range(len(outlierListIndecies)):
        data[outlierListIndecies[i]] = toReplace[i]
        
    return data


def findOutliers(data, bin2):
    # Calculate the standard deviation and mean of bin2
    stad = std(bin2)
    mean = np.mean(bin2)
    
    # Define the thresholds for outlier detection in bin2
    distancePos = 5 * stad + mean  # Upper threshold
    distanceNeg = (-5 * stad) + mean  # Lower threshold

    # Get the list of outliers and their corresponding indices in bin2
    (outlierList, outlierListIndecies) = getOutlierLists(bin2, distancePos, distanceNeg)
    
    if len(outlierList) == 0:
        # No outliers found, return the original data
        return data
    
    if min(outlierListIndecies) < 1350 and min(outlierListIndecies) > 90:
        # Special condition, return None
        return None
    
    # Calculate the standard deviation and mean of the original data
    stad = std(data)
    mean = np.mean(data)
    
    # Define the thresholds for outlier detection in the original data
    distancePos = 3 * stad + mean  # Upper threshold
    distanceNeg = (-3 * stad) + mean  # Lower threshold
    
    for o in range(len(outlierListIndecies)):
        if len(outlierListIndecies) < 5 or (o + 5 < len(outlierListIndecies) and np.abs(outlierListIndecies[o] - outlierListIndecies[o+5]) > 500 and outlierListIndecies[o] > 90):
            # Fill this single outlier and continue
            data[outlierListIndecies[o]] = findNearestValue(data, outlierListIndecies[o] - 1, outlierListIndecies[o], outlierListIndecies[o] + 1, distancePos, distanceNeg)
            continue
        elif outlierListIndecies[o] > 150:
            # This means the point is important, truncate the data
            data = data[:outlierListIndecies[o]]
            break
            
    return data


def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Designs a Butterworth bandpass filter.
    
    - Inputs:
        - lowcut: lower cutoff frequency of the filter.
        - highcut: upper cutoff frequency of the filter.
        - fs: sampling frequency of the signal.
        - order: order of the filter (default is 2).
        
    - Outputs:
        - b: numerator coefficients of the filter transfer function.
        - a: denominator coefficients of the filter transfer function.
    """
    
    nyq = 0.5 * fs  # Calculate the Nyquist frequency
    low = lowcut / nyq  # Normalize the lower cutoff frequency
    high = highcut / nyq  # Normalize the upper cutoff frequency
    
    # Adjust the stopband frequencies to avoid numerical instability issues
    high_s = min(high + 0.1, high + 4.000000000008001e-05)
    low_s = low - 0.1
    
    # Determine the filter order and the normalized frequencies for the design
    Ns, Wn = sig.buttord([low, high], [low_s, high_s], 3, 60)
    
    # Design the Butterworth bandpass filter
    b, a = sig.butter(2, [low, high], btype='band')
    
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    """
    Applies a Butterworth bandpass filter to the input data.
    
    - Inputs:
        - data: input data to be filtered.
        - lowcut: lower cutoff frequency of the filter.
        - highcut: upper cutoff frequency of the filter.
        - fs: sampling frequency of the data.
        - order: order of the filter (default is 6).
        
    - Outputs:
        - y: filtered data.
    """
    
    # Design the Butterworth bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    
    # Apply the filter to the data using the filter function
    y = sig.lfilter(b, a, data)
    
    return y


def butter_highpass_filter(data, cut, fs, order=2, axis_=0):
    """
    Applies a Butterworth highpass filter to the input data.
    
    - Inputs:
        - data: input data to be filtered.
        - cut: cutoff frequency of the highpass filter.
        - fs: sampling frequency of the data.
        - order: order of the filter (default is 2).
        - axis_: axis along which to apply the filter (default is 0).
        
    - Outputs:
        - y: filtered data.
    """
    
    nyq = 0.5 * fs  # Calculate the Nyquist frequency
    low = cut / nyq  # Normalize the cutoff frequency
    
    # Design the Butterworth highpass filter
    b, a = sig.butter(N=order, Wn=low, btype="high")
    
    # Apply the filter to the data using the lfilter function
    y = sig.lfilter(b, a, data, axis=axis_)
    
    return y


def find_peak_freqs(X, welch_cut_off):
    """
    Finds the peaks of the Welch signal, which will be used as an input to the self.extract_damp function for damping feature extraction.
    
    - Inputs:
        - X: Welch signal.
        - welch_cut_off: cutoff index for the Welch signal.
        
    - Outputs:
        - max_peaks: list of the three largest peak frequencies.
    """
    
    # Find peaks in the Welch signal
    peaks, _ = sig.find_peaks(X[welch_cut_off:])
    
    # Calculate peak prominences
    prominences = sig.peak_prominences(X[welch_cut_off:], peaks)
    
    # Find the three largest peak prominences
    max_vals = heapq.nlargest(3, prominences[0])
    
    # Get the corresponding peak frequencies for the largest peak prominences
    max_peaks = [peaks[np.where(prominences[0] == max_val)[0][0]] + welch_cut_off for max_val in max_vals]
    
    return max_peaks


def beamform(X,idx):
    """
    Performs beamforming on the input data.

    Parameters:
    - X: Input data of shape (# frames x # Rx x # Tx x # bins).
    - idx: Index parameter.

    Returns:
    - sph_pwr_max_loc: Maximum location of spherical power.
    - ph_bf: Unwrapped phase difference for beamforming.
    - sph_pwr_range_max_loc: Maximum location of spherical power range.
    """
    ph_bf = 0

    # Perform FFT on the input data
    beat_freq = scipy.fft.fft(X, axis=3)
    num_frms, tx, rx, N_range = beat_freq.shape

    # Define index parameters
    idxs = np.arange(idx-2, idx+3)

    # Define limits for theta and phi
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

    # Generate arrays for theta and phi
    theta = np.arange(theta_rad_lim[0], theta_rad_lim[1], d_theta)
    N_theta = len(theta)
    phi = np.arange(phi_rad_lim[0], phi_rad_lim[1], d_phi)
    N_phi = len(phi)

    lm = 3e8/77e9

    # Initialize arrays for spherical power
    sph_pwr = np.zeros((num_frms, N_theta, N_phi, 5), dtype=complex)
    sph_pwr_range = np.zeros((N_theta, N_phi, N_range), dtype=complex)

    x_idx = np.array([[0., 1., 2., 3.], [-2., -1., 0., 1.]])
    z_idx = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.]])
    s = lm / 2

    # Perform beamforming
    for kt in range(N_theta):
        for kp in range(N_phi):
            cos_theta = np.cos(theta[kt])
            sin_theta = np.sin(theta[kt])
            sin_phi = np.sin(phi[kp])

            sinp_cost = sin_phi * cos_theta
            sinp_sint = sin_phi * sin_theta

            Vec = np.exp(-1j*(2*np.pi*(s*z_idx*sinp_cost + s*x_idx*sinp_sint)/lm))
            VecRF = np.repeat(Vec[np.newaxis, :, :], num_frms, axis=0)
            VecRFR = np.repeat(VecRF[:, :, :, np.newaxis], N_range, axis=3)
            VecRFI = np.repeat(VecRF[:, :, :, np.newaxis], 5, axis=3)
            sph_pwr[:, kt, kp, :] = np.squeeze(np.sum(np.multiply(beat_freq[:, :, :, idxs], VecRFI), axis=(1, 2)))
            sph_pwr_range[kt, kp, :] = np.squeeze(np.sum(np.multiply(beat_freq, VecRFR), axis=(0, 1, 2)))

    # Calculate power and maximum locations
    pwr = np.squeeze(np.mean(abs(sph_pwr[:,:,:,2]),axis=0))**2  # Calculate the power by taking the mean of the absolute value of sph_pwr
    max_loc = np.unravel_index(pwr.argmax(), pwr.shape)  # Find the maximum location in the power array

    ph_bf = np.unwrap(np.angle(np.squeeze(sph_pwr[:,max_loc[0], max_loc[1],:])), axis=0)  # Unwrap the phase of the selected sph_pwr values
    ph_bf = ph_bf - np.mean(ph_bf,axis=0)  # Subtract the mean phase from the unwrapped phase array

    return sph_pwr[:,max_loc[0], max_loc[1],:], ph_bf, sph_pwr_range[max_loc[0], max_loc[1],:]  # Return the selected spherical power, unwrapped phase, and corresponding spherical power range




def gs_cofficient(v1, v2):
    """
    Returns the Gram-Schmidt coefficient between two vectors.

    Parameters:
    - v1: First vector.
    - v2: Second vector.

    Returns:
    - The Gram-Schmidt coefficient between v1 and v2.
    """
    return np.dot(v2, v1) / np.dot(v1, v1)


def multiply(cofficient, v):
    """
    Multiplies each element of the vector v by the given coefficient.

    Parameters:
    - cofficient: Coefficient to multiply the vector elements with.
    - v: Vector to be multiplied.

    Returns:
    - The resulting vector after multiplying each element by the coefficient.
    """
    return map((lambda x: x * cofficient), v)


def proj(v1, v2):
    """
    Calculates the projection of vector v2 onto vector v1 using the Gram-Schmidt process.
    
    Parameters:
    - v1: First vector.
    - v2: Second vector.
    
    Returns:
    - Projection of v2 onto v1.
    """
    return multiply(gs_cofficient(v1, v2), v1)


def GramSchmidt(X):
    """
    Applies the Gram-Schmidt process to the vectors in the input matrix X, producing an orthogonal basis.
    
    Parameters:
    - X: Input matrix of shape (n, m), where n is the number of vectors and m is the dimension of each vector.
    
    Returns:
    - Y: Orthogonal basis formed by the vectors after applying the Gram-Schmidt process.
    """
    Y = []
    for i in range(X.shape[0]):
        temp_vec = X[i]
        for inY in Y:
            proj_vec = proj(inY, X[i])
            # Compute the projection of the current vector onto the previously processed vectors
            temp_vec = map(lambda x, y: x - y, temp_vec, proj_vec)
            # Subtract the projections from the current vector to remove components along the previous vectors
        Y.append(temp_vec)
    return Y


def compensate_range_bins(base_bins, obj_bins):
    """
    Applies range bin compensation to the object bins using the base bins.
    
    Parameters:
    - base_bins: Matrix of base bins with an orthogonal basis, of shape (n, m), where n is the number of base bins and m is the dimension of each bin.
    - obj_bins: Matrix of object bins to be compensated, of shape (p, m), where p is the number of object bins and m is the dimension of each bin.
    
    Returns:
    - y: Matrix of compensated object bins after applying range bin compensation, of shape (p, m).
    """
    base_bins = GramSchmidt(base_bins)
    y = obj_bins
    
    # Iterate over each object bin
    for ii in range(obj_bins.shape[1]):
        # Iterate over each base bin
        for jj in range(base_bins.shape[1]):
            # Compute the dot product between the base bin and object bin
            dot_prod = np.dot(base_bins[:, jj], obj_bins[:, ii])
            
            # Subtract the projection of the base bin onto the object bin from the object bin
            y[:, ii] = y[:, ii] - dot_prod * base_bins[:, jj]
    
    return y

    

def isNotOutlier(point, upper, lower):
    """
    Checks if a given point is not an outlier, based on upper and lower thresholds.

    Parameters:
    - point: The point to be checked.
    - upper: The upper threshold.
    - lower: The lower threshold.

    Returns:
    - True if the point is not an outlier, False otherwise.
    """
    return (point < upper and point > lower)


def findNearestValue(data, before, current, after, threshAbove, threshBelow):
    """
    Finds the nearest value in the given data that is not an outlier, considering the specified thresholds.

    Parameters:
    - data: The data array.
    - before: Index of the value before the current value.
    - current: Index of the current value.
    - after: Index of the value after the current value.
    - threshAbove: Upper threshold for outlier detection.
    - threshBelow: Lower threshold for outlier detection.

    Returns:
    - The nearest value that is not an outlier, or None if no such value is found.
    """
    # Check if before and after indices are within the valid range
    before = before if before > -1 else 0
    after = after if after < len(data) else len(data) - 1

    while True:
        # Check if the value after the current value is within the data range and not an outlier
        if after < len(data) and isNotOutlier(data[after], threshAbove, threshBelow):
            return data[after]
        after += 1

        # Check if the value before the current value is within the data range and not an outlier
        if before >= 0 and isNotOutlier(data[before], threshAbove, threshBelow):
            return data[before]
        before -= 1

        # If both before and after indices are out of range, return None
        if after > len(data) and before < 0:
            return None


def getOutlierLists(data, distancePos, distanceNeg):
    """
    Returns the lists of outliers and their corresponding indices in the given data, based on positive and negative distance thresholds.

    Parameters:
    - data: The data array.
    - distancePos: The positive distance threshold for outlier detection.
    - distanceNeg: The negative distance threshold for outlier detection.

    Returns:
    - outlierList: List of outlier values.
    - outlierListIndices: List of indices corresponding to the outlier values.
    """
    # Initialize an empty list to store the outlier values
    outlierList = []

    # Extend the outlierList with values greater than distancePos and less than distanceNeg
    outlierList.extend(data[data > distancePos].tolist())
    outlierList.extend(data[data < distanceNeg].tolist())

    # Find the indices of the outlier values in the original data
    outlierListIndices = [i for i, j in enumerate(data) if j in outlierList]

    return outlierList, outlierListIndices



def findOutliers(data, threshold):
    """
    Finds the outliers in the given data array based on the specified threshold.

    Parameters:
    - data: The data array.
    - threshold: The threshold for outlier detection.

    Returns:
    - indices: NumPy array of indices corresponding to the outliers.
    """
    indices = []  # Initialize an empty list to store the indices of outliers
    stad = std(data)  # Calculate the standard deviation of the data

    if(stad > 0.009):  # Check if the standard deviation exceeds a certain threshold
        median = np.median(data)  # Calculate the median of the data
        MAD = np.median(np.absolute(data - np.median(data)))  # Calculate the Median Absolute Deviation (MAD)

        # Calculate the positive and negative distance thresholds
        distancePos = threshold * MAD + median
        distanceNeg = (-threshold * MAD) + median

        # Call the getOutlierLists function to obtain the outlier list and their indices
        (outlierList, indices) = getOutlierLists(data, distancePos, distanceNeg)

    # Convert the list of indices to a NumPy array
    return np.asarray(indices)

    
def softmax(x, t):
    """
    Computes the softmax function for the input array.

    Parameters:
    - x: Input array.
    - t: Temperature parameter.

    Returns:
    - Softmax output.
    """
    return np.exp(x) / np.sum(np.exp(x), 0)


def rolling_std(a, n=3):
    """
    Calculates the rolling standard deviation of an array.

    Parameters:
    - a: Input array.
    - n: Window size for rolling calculation.

    Returns:
    - Rolling standard deviation array.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def outliers_exist(x, threshold):
    """
    Checks if outliers exist in the input array based on the specified threshold.

    Parameters:
    - x: Input array.
    - threshold: Threshold for outlier detection.

    Returns:
    - Boolean value indicating if outliers exist.
    """
    std = np.std(x)  # Calculate the standard deviation of the array

    if std > threshold:  # Check if the standard deviation exceeds the threshold
        return True

    return False


def filloutliers(data_2, data, threshold):
    """
    Fills outliers in the input data array with nearest values.

    Parameters:
    - data_2: Input data array.
    - data: Original data array.
    - threshold: Threshold for outlier detection.

    Returns:
    - Number of outliers filled and the updated data array.
    """
    data_2 = np.squeeze(data_2)  # Remove single-dimensional entries from the shape of data_2
    data = np.squeeze(data)  # Remove single-dimensional entries from the shape of data
    outlierListIndecies = []  # Initialize an empty list to store outlier indices

    if outliers_exist(data_2, 0.009):  # Check if outliers exist in data_2
        median = np.median(data_2)  # Calculate the median of data_2
        MAD = np.median(np.absolute(data_2 - np.median(data_2)))  # Calculate the Median Absolute Deviation (MAD) of data_2
        distancePos = threshold * MAD + median  # Calculate the upper threshold for outlier detection
        distanceNeg = (-threshold * MAD) + median  # Calculate the lower threshold for outlier detection

        outlierList, outlierListIndecies = getOutlierLists(data_2, distancePos, distanceNeg)  # Get outlier list and indices

        toReplace = []  # Initialize an empty list to store replacement values for outliers

        # Iterate over each outlier index
        for i in range(len(outlierList)):
            # Find the nearest value to replace the outlier by searching in the neighboring indices
            toReplace.append(findNearestValue(data_2, outlierListIndecies[i] - 1, outlierListIndecies[i],
                                              outlierListIndecies[i] + 1, distancePos, distanceNeg))

        # Iterate over each outlier index
        for i in range(len(outlierListIndecies)):
            # Replace the outlier in data_2 with the corresponding replacement value
            data_2[outlierListIndecies[i]] = toReplace[i]

    return len(outlierListIndecies), data_2  # Return the number of outliers filled and the updated data_2 array


def clean_data(X, bin_idx):
    """
    Cleans the data by removing outliers from the phase values.

    Parameters:
    - X: Input data dictionary.
    - bin_idx: Index of the bin to clean.

    Returns:
    - Number of non-outliers, number of outliers, and the cleaned phase data.
    """
    phase2 = X['X_phase_all']  # Extract phase data from input dictionary
    phase2 = phase2[0:2500, :]  # Take the first 2500 samples to include at least 9 pulses and reduce outliers
    filt_phase_all2 = butter_highpass_filter(phase2, 15, 250, axis_=0)  # Apply a high-pass filter to remove low-frequency offsets
    num_outliers, phase_cleaned = filloutliers(filt_phase_all2[:, 1], filt_phase_all2[:, bin_idx], threshold=8)
    # Remove outliers from the specified bin using the filloutliers function

    num_nonoutliers = phase2.shape[0] - num_outliers  # Calculate the number of non-outliers

    return num_nonoutliers, num_outliers, phase_cleaned
    # Return the number of non-outliers, number of outliers, and the cleaned phase data

    
### end helper functions ###

def extract_rf_features(X, range_res):
    """
    Extracts RF features from the input data.

    Parameters:
    - X: Input data of size (num_frames x 64).
    - range_res: Range resolution.

    Returns:
    - mrf_features_squared: RF features multiplied by distance squared.
    - mrf_features: RF features multiplied by distance.
    - power_features: Power features.
    """
    X = np.abs(np.sum(X, axis=0))  # Sum the input data along the frames axis and take the absolute value
    max_idxs = np.arange(0, 64)  # Create an array of indices from 0 to 63
    mrf_features_squared = X * (range_res * max_idxs) ** 2  # Multiply the RF features by the square of the distance
    mrf_features = X * (range_res * max_idxs)  # Multiply the RF features by the distance
    power_features = X  # Assign the power features as the absolute sum of the input data

    return mrf_features_squared, mrf_features, power_features
    # Return the RF features multiplied by distance squared, RF features multiplied by distance, and power features


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