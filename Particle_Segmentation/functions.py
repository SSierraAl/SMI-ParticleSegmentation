
from libraries_Import import *  # Essential libraries for data processing

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
