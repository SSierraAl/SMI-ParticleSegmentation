
import numpy as np
from scipy.signal import butter, filtfilt,spectrogram, hilbert
import pandas as pd
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot  as plt
# Fast Fourier Transform (FFT) calculation
################################################################################
def FFT_calc(datos, samplefreq):
    """
    Calculate the FFT (Fast Fourier Transform) of the input data.
    
    Parameters:
    datos : array-like
        Time-domain signal data.
    samplefreq : int
        Sampling frequency of the data in Hz.

    Returns:
    amplitude : array-like
        Amplitude spectrum of the signal.
    freq_fft : array-like
        Corresponding frequency values.
    phase : array-like
        Phase spectrum of the signal.
    """
    n = len(datos)  # Length of the data
    # Perform FFT and calculate amplitude and phase
    datos = np.asarray(datos, dtype=np.float64)
    fft_result = np.fft.rfft(datos)  # Compute FFT
    freq_fft = np.fft.rfftfreq(len(datos), 1 / samplefreq)  # Corresponding frequencies
    amplitude = np.abs(fft_result)  # Magnitude of the FFT
    phase = np.angle(fft_result)  # Phase of the FFT

    return amplitude, freq_fft, phase

# Butterworth bandpass filter
################################################################################
def butter_bandpass_filter(data, lowcut, highcut, order, fs):
    """
    Apply a Butterworth bandpass filter to the input data.

    Parameters:
    data : array-like
        Input signal data to be filtered.
    lowcut : float
        Lower cutoff frequency for the filter.
    highcut : float
        Upper cutoff frequency for the filter.
    order : int
        Order of the filter.
    fs : float
        Sampling frequency of the input data.

    Returns:
    y : array-like
        Filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency (half of the sampling frequency)
    
    # Normalize the cutoff frequencies by the Nyquist frequency
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    
    # Design a Butterworth bandpass filter
    b, a = butter(order, [lowcut, highcut], btype='band', analog=False)
    
    # Apply the filter to the data using filtfilt (zero-phase filtering)
    y = filtfilt(b, a, data)
    
    return y

# Gaussian function for curve fitting
################################################################################
def gaussian(x, amplitude, mean, stddev):
    """
    Define a Gaussian function for curve fitting.

    Parameters:
    x : array-like
        Independent variable (usually time or frequency).
    amplitude : float
        Amplitude of the Gaussian peak.
    mean : float
        Mean (center) of the Gaussian peak.
    stddev : float
        Standard deviation (spread) of the Gaussian peak.

    Returns:
    array-like
        Gaussian function values.
    """
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

# Spectrogram-based particle analysis
################################################################################
def find_spectogram(data, Adq_Freq, FFT_Amp, FFT_Freq, Particle_Params):
    """
    Generate an adaptive spectrogram based on particle detection parameters.

    Parameters:
    data : array-like
        Time-domain signal data.
    Adq_Freq : float
        Acquisition frequency of the data (in Hz).
    FFT_Amp : array-like
        Amplitude of the FFT of the signal.
    FFT_Freq : array-like
        Frequency values corresponding to the FFT amplitudes.
    Particle_Params : dict
        Parameters related to particle properties, including laser wavelength, angle, and beam spot size.

    Returns:
    t : array-like
        Time values for the spectrogram.
    f_new : array-like
        Filtered frequency values of the spectrogram.
    Sxx_new : 2D array
        Spectrogram values corresponding to filtered frequencies.
    """
    # Frequency range to analyze around the Doppler peak
    range_freq = 8000  # Range of frequency (in Hz) around the Doppler peak

    # Particle and laser parameters
    laser_lambda = Particle_Params['laser_lambda']
    angle = Particle_Params['angle']
    sin_value = np.sin(np.deg2rad(angle))  # Sine of the angle in radians
    distance = Particle_Params['beam_spot_size']

    # Find the Doppler peak (maximum peak in the FFT)
    index_max_peak = np.argmax(FFT_Amp)  # Index of the max FFT amplitude
    doppler_peak = round(FFT_Freq[index_max_peak])  # Doppler frequency

    # Calculate frequency range for analysis
    fc1 = max(doppler_peak - range_freq, 0)  # Lower cutoff frequency
    fc2 = doppler_peak + range_freq  # Upper cutoff frequency

    # Calculate the speed of the particle using Doppler shift
    fd = doppler_peak
    speed = ((fd * laser_lambda) / (abs(sin_value) * 2))

    # Estimate the interaction time of the particle with the laser beam (Tao)
    Tao = distance / speed

    # Calculate segment size for the spectrogram
    new_nperseg = round((Tao / 10) * Adq_Freq / 2) * 2  # Ensure segment size is even
    tiempo = np.arange(0, len(data), 1) / 1000  # Time axis in milliseconds

    # Generate the spectrogram with Blackman window
    noverlap = new_nperseg / 2
    f, t, Sxx = spectrogram(data, fs=Adq_Freq, window='blackman', nperseg=new_nperseg, noverlap=noverlap)

    # Filter the spectrogram within the frequency range of interest
    f_idx = (f >= fc1) & (f <= fc2)
    f_new = f[f_idx] / 1000  # Convert frequency to kHz
    Sxx_new = Sxx[f_idx, :]  # Filtered spectrogram

    return t, f_new, Sxx_new

# Classification of zones in the signal based on value thresholds
################################################################################
def classify_zones(data, length_range):
    """
    Classify signal zones based on their values compared to the mean value of the data.

    Parameters:
    data : array-like
        Input signal data.
    length_range : list
        List containing the minimum and maximum length of a valid zone.

    Returns:
    valid_zones : list of tuples
        List of valid zones (start and end indices).
    anomalies : list of tuples
        List of anomalous zones (start and end indices).
    """
    mean_value = sum(data) / len(data)  # Calculate the mean value of the signal
    valid_zones = []
    anomalies = []
    current_zone = []

    # Loop through the data to find valid zones
    for i, value in enumerate(data):
        if value > mean_value:
            current_zone.append(i)  # Append index to current zone if value is above the mean
        else:
            # End the current zone when value drops below the mean
            if current_zone:
                # Check if the zone length is within the specified range
                if length_range[0] <= len(current_zone) <= length_range[1] and all(data[j] > mean_value for j in current_zone):
                    valid_zones.append((current_zone[0], current_zone[-1]))  # Add valid zone
                else:
                    anomalies.append((current_zone[0], current_zone[-1]))  # Add anomalous zone
                current_zone = []

    # Check the last zone after the loop ends
    if current_zone:
        if length_range[0] <= len(current_zone) <= length_range[1] and all(data[j] > mean_value for j in current_zone):
            valid_zones.append((current_zone[0], current_zone[-1]))
        else:
            anomalies.append((current_zone[0], current_zone[-1]))

    return valid_zones, anomalies

# Mask adjustment for particle extraction
################################################################################
def adjust_mask_and_extract(data_X, highlight_mask, desired_length=1970):
    """
    Adjust the highlight mask to a specific length for particle extraction.

    Parameters:
    data_X : array-like
        Time axis data corresponding to the signal.
    highlight_mask : array-like
        Boolean mask indicating highlighted (True) regions of interest.
    desired_length : int, optional
        The desired number of points to extract (default is 1970).

    Returns:
    adjusted_mask : array-like
        The adjusted mask for particle extraction with the desired length.
    """
    if highlight_mask.sum() == 0:
        raise ValueError("The highlight mask contains no True values.")  # Ensure mask is not empty
    
    # Find the indices of True values in the mask
    true_indices = np.where(highlight_mask)[0]
    start, end = true_indices[0], true_indices[-1]  # Determine the start and end indices
    
    current_length = end - start + 1  # Calculate the current length of the masked region

    # Adjust the mask to fit the desired length
    if current_length > desired_length:
        # Shrink the mask to fit the desired length
        center = (start + end) // 2
        start = center - desired_length // 2
        end = start + desired_length - 1
    else:
        # Expand the mask to fit the desired length
        additional_points = desired_length - current_length
        start -= additional_points // 2
        end = start + desired_length - 1

    # Handle boundary conditions to ensure valid indices
    if start < 0:
        start = 0
        end = desired_length - 1
    if end >= len(data_X):
        end = len(data_X) - 1
        start = end - desired_length + 1

    # Create the adjusted mask
    adjusted_mask = np.zeros_like(highlight_mask, dtype=bool)
    adjusted_mask[start:end + 1] = True  # Set the adjusted mask range to True

    return adjusted_mask




#   Load multiple files
################################################################################
def  Load_New_Data(filepath, newdata):
    if newdata==True:
        folder_path = filepath
        file_names = sorted(os.listdir(folder_path))
        data_list = []
        del file_names[0]
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            data_list.append(data)
        #np.save('outputConcatenatedData.npy', data_list)
        df = pd.DataFrame(data_list)

        Data_Frame= df.transpose()
        num_cols = Data_Frame.shape[1]
        interval = 10

        for i in range(num_cols):
            pos = i // interval
            exp = i % interval
            col_name = f'Pos{pos}_exp{exp}'
            col_name=file_names[i]
            Data_Frame.rename(columns={i: col_name}, inplace=True)
        Data_Frame.to_csv('DataframewithOrder.csv', index=False)

    else:
        Data_Frame = pd.read_csv('./DataframewithOrder.csv')

    return Data_Frame

# Particle and Anomalies Validation ###########################################

def particle_anomalies_validation(anomalies,valid_zones,data_X,y_Filtrada,t,line_Graph_Line, Green_Group, Yellow_Group, Red_Group):

    if anomalies: 
        for aa in anomalies:
            aa=list(aa)
            if aa !=(0,0):
                if aa[0]<3:
                    aa[0]=3
                if aa[1]+3>=len(t):
                    aa[1]=-4
                # Add Span for start
                highlight_mask = (data_X >= t[aa[0]-3]) & (data_X <= t[aa[1]+3])

                highlight_mask=adjust_mask_and_extract(data_X, highlight_mask, 2500)

                valid_P_x=data_X[highlight_mask]
                valid_P_y=y_Filtrada[highlight_mask]

                x_data=np.linspace(0, len(valid_P_x), len(valid_P_x))
                valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                # Step 2: Scale to -1 to 1
                valid_P_y2 = 2 * valid_P_y2 - 1
                valid_P_y2=valid_P_y-np.mean(valid_P_y)
                # Ajuste de curva utilizando la función gaussiana con amplitud inicial más alta
                valid_P_y2 = np.abs(hilbert(valid_P_y))
                try:
                    optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                    # Obtener los parámetros optimizados
                    amplitude, mean, stddev = optimized_params
                    amplitude = amplitude/ np.max((amplitude))
                    amplitude=amplitude*max(valid_P_y)
                    y_curve = gaussian(x_data, amplitude, mean, stddev)
                    if (max(y_curve)> y_curve[0]+0.05) and (max(y_curve)> y_curve[-1]+0.05):
                        line_Graph_Line.line(valid_P_x, valid_P_y, line_width=2, line_color="yellow", legend_label="P_anomaly_fit")
                        line_Graph_Line.line(valid_P_x, y_curve, line_width=2, line_color="yellow")
                        Yellow_Group=Yellow_Group+1
                    else:
                        line_Graph_Line.line(valid_P_x, valid_P_y, line_width=2, line_color="red", legend_label="P_anomaly_fit")
                        Yellow_Group=Yellow_Group+1
                        #filterImpact.line(valid_P_x, y_curve, line_width=2, line_color="red", line_alpha=0.5)
                except:
                    line_Graph_Line.line(valid_P_x, valid_P_y, line_width=2, line_color="red", legend_label="P_anomaly")
                    Red_Group=Red_Group+1

    if valid_zones: 
        for vv in valid_zones:
            vv=list(vv)
            if vv != (0,0):
                if vv[0]<3:
                    vv[0]=3
                if vv[1]+3>=len(t):
                    vv[1]=-4
                # Add Span for start
                highlight_mask = (data_X >= t[vv[0]-3]) & (data_X <= t[vv[1]+3])

                highlight_mask=adjust_mask_and_extract(data_X, highlight_mask, 2500)

                valid_P_x=data_X[highlight_mask]
                valid_P_y=y_Filtrada[highlight_mask]
                #filterImpact.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="Valid_P")
                x_data=np.linspace(0, len(valid_P_x), len(valid_P_x))
                valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                # Step 2: Scale to -1 to 1
                valid_P_y2 = 2 * valid_P_y2 - 1
                valid_P_y2=valid_P_y-np.mean(valid_P_y)
                # Ajuste de curva utilizando la función gaussiana con amplitud inicial más alta
                valid_P_y2 = np.abs(hilbert(valid_P_y))
                initial_params = [np.max(valid_P_y2)*10, np.argmax(valid_P_y2), 1.0]
                try:
                    optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                    # Obtener los parámetros optimizados
                    amplitude, mean, stddev = optimized_params
                    amplitude = amplitude/ np.max((amplitude))
                    amplitude=amplitude*max(valid_P_y)
                    y_curve = gaussian(x_data, amplitude, mean, stddev)
                    if (max(y_curve)> y_curve[0]+0.05) and (max(y_curve)> y_curve[-1]+0.05):
                        line_Graph_Line.line(valid_P_x, valid_P_y, line_width=2, line_color="green", legend_label="Valid_P")
                        line_Graph_Line.line(valid_P_x, y_curve, line_width=2, line_color="green")
                        Green_Group=Green_Group+1
                    else:
                        line_Graph_Line.line(valid_P_x, valid_P_y, line_width=2, line_color="yellow", legend_label="P_No_Fit")
                        Yellow_Group=Yellow_Group+1
                        #filterImpact.line(valid_P_x, y_curve, line_width=2, line_color="yellow",line_alpha=0.5)
                except:
                    line_Graph_Line.line(valid_P_x, valid_P_y, line_width=2, line_color="yellow", legend_label="P_No_Fit")
                    Yellow_Group=Yellow_Group+1

    return Green_Group, Yellow_Group, Red_Group

    #########################################################
    ##########################################################


    # Particle Estimation ###########################################
def particle_estimation(FFT_Amp,FFT_Freq,Particle_Params, timeZone,Adq_Freq):
    
    laser_lambda=Particle_Params['laser_lambda']
    angle=Particle_Params['angle']
    distance=Particle_Params['beam_spot_size']

    doppler_peak = np.argmax(FFT_Amp)
    doppler_peak = round(FFT_Freq[doppler_peak])

    speed=((doppler_peak*laser_lambda)/(abs(np.sin(np.deg2rad(angle)))*2))

    time= (timeZone/1000)/speed


    return doppler_peak, speed, time




#FWHW ###############################################################################
def get_full_width(x: np.ndarray, y: np.ndarray, height: float = 0.5) -> float:
    height_half_max = np.max(y) * height
    index_max = np.argmax(y)
    x_low = np.interp(height_half_max, y[:index_max+1], x[:index_max+1])
    x_high = np.interp(height_half_max, np.flip(y[index_max:]), np.flip(x[index_max:]))

    return x_high, x_low





#Load and concatenate multiples files
def GetLoad_Folder(directory,name_especific, new_data):
    print(name_especific)
    if new_data==True:
        data = []        
        npy_files = [file for file in os.listdir(directory) if file.endswith('.npy')]
        print("Number of files")
        print(len(npy_files))
        print('-----------------')
        # Read each .npy file and append the data to the list
        for file in npy_files:
            file_path = os.path.join(directory, file)
            file_data = np.load(file_path)
            data.append(file_data)
        # Convert the list of data into a Numpy array
        data = np.array(data,dtype=object)
        np.save(('./Temporal_Files/'+name_especific+'.npy'), data)
    else:
        data = np.load('./Temporal_Files/'+name_especific+'.npy')
    return data



# average FFT 
def AverageFFT(data_array,freq_cut_inf, freq_cut_sup, filterorder, samplefreq):
    freq_ind=0
    Freq_Data=pd.DataFrame()
    for i, dataset in enumerate(data_array):
        dataset= butter_bandpass_filter(dataset, freq_cut_inf, freq_cut_sup, filterorder, samplefreq)
        amp, freq_ind2, phase = FFT_calc(dataset, samplefreq)
        # Ajuste solo por un lio con los archivos de particulas de 4um
        if(len(freq_ind2)==950):
            pass
        else:
            freq_ind=freq_ind2
            amp=pd.Series(amp)
            Freq_Data=pd.concat([Freq_Data, amp], axis=1)

    DataAVG=(Freq_Data.mean(axis=1))

    return DataAVG, freq_ind



def plot_grouped_error_bars(filename, num_groups):
    # Load the CSV file, assuming the first column is an index
    df = pd.read_csv(filename, index_col=0)
    
    # Invert the values in the DataFrame
    df = df.apply(lambda row: row.max() + row.min() - row, axis=1)
        #df[column] = max_val + min_val - df[column]
    
    # Calculate mean and standard deviation for each column after inversion
    means = df.mean()
    std_devs = df.std()

    # Determine the number of columns per group
    num_columns = len(df.columns)
    columns_per_group = num_columns // num_groups
    remainder = num_columns % num_groups

    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 8))  # Adjust the figure size as necessary
    fig.suptitle('Inverted Mean Values with Standard Deviation Error Bars by Group')

    for i in range(num_groups):
        start_idx = i * columns_per_group
        if i == num_groups - 1:  # Last group takes the remainder
            end_idx = start_idx + columns_per_group + remainder
        else:
            end_idx = start_idx + columns_per_group

        group_means = means[start_idx:end_idx]
        group_stds = std_devs[start_idx:end_idx]
        x = np.arange(len(group_means))  # the label locations for this subgroup

        # Plotting for this subgroup
        axes[i].errorbar(x, group_means, yerr=group_stds, fmt='o', ecolor='red', capthick=2, alpha=0.6, label='Mean ± 1 SD')
        axes[i].plot(x, group_means, 'bo-', label='Mean Trendline')  # Line connecting the means
        axes[i].set_ylabel('Values')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(df.columns[start_idx:end_idx])
        axes[i].legend()

        for tick in axes[i].get_xticklabels():
            tick.set_rotation(45)

    plt.xlabel('Columns')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the global title
    plt.show()
