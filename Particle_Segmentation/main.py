
from libraries_Import import *  # Essential libraries for data processing
from functions import *  # Functions to handle signal processing tasks

# Define the path to the folder containing particle data
path_folder = "./Particles_Data/"

# Names of folders containing raw particle data
Name_Folder = ['Raw_2um',  # Data for particles with a size of 2 micrometers
               'Raw_4um',  # Data for particles with a size of 4 micrometers
               'Raw_10um']  # Data for particles with a size of 10 micrometers

# Corresponding names of folders where processed data will be saved
Final_Folder = ['DB_10um',  # Folder for processed data of 10 micrometer particles
                'DB_4um',   # Folder for processed data of 4 micrometer particles
                'DB_2um']   # Folder for processed data of 2 micrometer particles

# Define experiment parameters
# Acquisition frequency in Hertz (samples per second)
Adq_Freq = 2000000  
# Time duration of captured data in milliseconds
Time_Captured_ms = 16384  
# Parameters of the SMI sensor used for particle detection
Particle_Params = {
    "laser_lambda": 1.55e-6,  # Wavelength of the laser in meters
    "angle": 22,              # Angle of incidence in degrees
    "beam_spot_size": 90e-6,  # Size of the laser beam spot in meters
}

# Function to process data from a file, extract valuable signal sections, and save them
def Get_valuable_section(data_Y, dst_folder, filename, Adq_Freq, Time_Captured_ms, Particle_Params):
    
    global counter_counter
    # Initialize a counter for processed files
    global Counter_Column
    Counter_Column = 1

    # Convert data to a pandas series for easier manipulation
    data_Y = pd.Series(data_Y)
    # Calculate the time axis (X-axis) based on captured time and acquisition frequency
    data_X_max = Time_Captured_ms * 1000 / Adq_Freq
    data_X = np.linspace(0, data_X_max, Time_Captured_ms)

    # Remove the DC offset (mean) from the signal
    data_Y_NoOffset = data_Y - data_Y.mean()

    # Apply bandpass filtering to the data to isolate the desired frequency range
    y_Filtrada = butter_bandpass_filter(data_Y_NoOffset, 7000, 100000, 4, Adq_Freq)

    # Perform FFT (Fast Fourier Transform) on the filtered signal
    ampFFT1, freqfft1, _ = FFT_calc(y_Filtrada, Adq_Freq)

    # Spectrogram analysis to find frequency peaks
    std_dev = 20  # Standard deviation for Gaussian filtering
    cut_index = int(len(ampFFT1) / 10)  # Analyze only the lower frequency range
    new_FFT = ampFFT1[0:cut_index]
    filtered_fft = gaussian_filter1d(new_FFT, sigma=std_dev)

    # Find peaks in the filtered FFT signal
    mean_amplitude = (max(filtered_fft) / 2)
    peaks, properties = find_peaks(filtered_fft, height=mean_amplitude)
    peak_amplitudes = properties["peak_heights"]

    # Sort peaks by amplitude and apply a minimum distance criterion (12 kHz)
    sorted_indices = np.argsort(peak_amplitudes)[::-1]  # Sort in descending order of amplitude
    sorted_peaks = peaks[sorted_indices]
    min_distance = 12000  # Minimum frequency separation between peaks in Hz
    valid_peaks = []

    # Ensure valid peaks are spaced by at least 12 kHz
    for peak in sorted_peaks:
        if all(abs(freqfft1[peak] - freqfft1[vp]) >= min_distance for vp in valid_peaks):
            valid_peaks.append(peak)
    
    # Generate a spectrogram for the filtered data
    t, f_new, Sxx_new = find_spectogram(y_Filtrada, Adq_Freq, ampFFT1, freqfft1, Particle_Params)

    # Compute the mean of the spectrogram and classify valid signal zones and anomalies
    general_mean = np.mean(Sxx_new, axis=0)
    t = np.linspace(0, max(data_X), len(general_mean))  # Adjust the time axis for plotting
    valid_zones, anomalies = classify_zones(general_mean, [5, 10])

    # Process valid signal zones and fit them to a Gaussian curve
    if valid_zones: 
        for vv in valid_zones:
            vv = list(vv)
            if vv != (0, 0):
                # Adjust start and end points to include a buffer
                if vv[0] < 3:
                    vv[0] = 3
                if vv[1] + 3 >= len(t):
                    vv[1] = -4

                # Extract and scale the valid section of the signal
                highlight_mask = (data_X >= t[vv[0] - 3]) & (data_X <= t[vv[1] + 3])
                highlight_mask = adjust_mask_and_extract(data_X, highlight_mask, 2500)
                valid_P_x = data_X[highlight_mask]
                valid_P_y = y_Filtrada[highlight_mask]

                # Normalize the signal for Gaussian fitting
                x_data = np.linspace(0, len(valid_P_x), len(valid_P_x))
                valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                valid_P_y2 = 2 * valid_P_y2 - 1
                valid_P_y2 = valid_P_y - np.mean(valid_P_y)
                valid_P_y2 = np.abs(hilbert(valid_P_y))

                try:
                    # Perform Gaussian fitting on the valid signal
                    optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                    amplitude, mean, stddev = optimized_params
                    amplitude = amplitude / np.max((amplitude))  # Normalize amplitude
                    amplitude = amplitude * max(valid_P_y)
                    y_curve = gaussian(x_data, amplitude, mean, stddev)

                    # Save the valid section if Gaussian fitting is successful
                    if (max(y_curve) > y_curve[0] + 0.05) and (max(y_curve) > y_curve[-1] + 0.05):
                        save_path = os.path.join(dst_folder, filename + str(counter_counter))
                        np.save(save_path, valid_P_y)
                        print(f"File {filename} processed and saved to {dst_folder}")
                        counter_counter += 1
                except:
                    print(f' --- Error in Gaussian fitting --- {filename}')

    # Additional processing if more than one particle was detected
    if len(valid_peaks) > 1:
        two_peak = int(freqfft1[valid_peaks][1])
        min_lim = max(two_peak - 12000, 7000)  # Set a lower limit for frequency filtering
        max_lim = two_peak + 12000  # Set an upper limit for frequency filtering
        print('filter limits')
        print(min_lim)    
        print(max_lim)

        # Re-filter the signal and repeat the process
        data_Y_NoOffset = data_Y - data_Y.mean()
        y_Filtrada = butter_bandpass_filter(data_Y_NoOffset, min_lim, max_lim, 4, Adq_Freq)
        ampFFT1, freqfft1, _ = FFT_calc(y_Filtrada, Adq_Freq)
        t, f_new, Sxx_new = find_spectogram(y_Filtrada, Adq_Freq, ampFFT1, freqfft1, Particle_Params)
        general_mean = np.mean(Sxx_new, axis=0)
        t = np.linspace(0, max(data_X), len(general_mean))
        valid_zones, anomalies = classify_zones(general_mean, [5, 10])
        print(valid_zones)
        print(anomalies)

        # Repeat Gaussian fitting and saving process for valid zones
        if valid_zones: 
            for vv in valid_zones:
                vv = list(vv)
                if vv != (0, 0):
                    if vv[0] < 3:
                        vv[0] = 3
                    if vv[1] + 3 >= len(t):
                        vv[1] = -4

                    highlight_mask = (data_X >= t[vv[0] - 3]) & (data_X <= t[vv[1] + 3])
                    highlight_mask = adjust_mask_and_extract(data_X, highlight_mask, 2500)
                    valid_P_x = data_X[highlight_mask]
                    valid_P_y = y_Filtrada[highlight_mask]
                    x_data = np.linspace(0, len(valid_P_x), len(valid_P_x))
                    valid_P_y2 = (valid_P_y - np.min(valid_P_y)) / (np.max(valid_P_y) - np.min(valid_P_y))
                    valid_P_y2 = 2 * valid_P_y2 - 1
                    valid_P_y2 = valid_P_y - np.mean(valid_P_y)
                    valid_P_y2 = np.abs(hilbert(valid_P_y))

                    try:
                        optimized_params, _ = curve_fit(gaussian, x_data, valid_P_y2)
                        amplitude, mean, stddev = optimized_params
                        amplitude = amplitude / np.max((amplitude))
                        amplitude = amplitude * max(valid_P_y)
                        y_curve = gaussian(x_data, amplitude, mean, stddev)
                        if (max(y_curve) > y_curve[0] + 0.05) and (max(y_curve) > y_curve[-1] + 0.05):
                            save_path = os.path.join(dst_folder, filename + str(counter_counter))
                            np.save(save_path, valid_P_y)
                            print(f"File {filename} processed and saved to {dst_folder}")
                            counter_counter += 1
                    except:
                        print(f' --- Error in Gaussian fitting --- {filename}')


# Main function to process files from source folders and save results to destination folders
def process_and_save_files(path_folder, source_folders, destination_folders):
    global counter_counter
    counter_counter = 0

    # Ensure source and destination folder lists are of the same length
    if len(source_folders) != len(destination_folders):
        raise ValueError("Source and destination folders lists must have the same length.")

    # Update paths for source folders
    src_folder_new = [path_folder + i for i in source_folders]
    source_folders = src_folder_new

    # Update paths for destination folders
    destination_folders_new = [path_folder + i for i in destination_folders]
    destination_folders = destination_folders_new

    # Process each file in each source folder
    for src_folder, dst_folder in zip(source_folders, destination_folders):
        # Ensure destination folder exists
        os.makedirs(dst_folder, exist_ok=True)

        # Loop over all files in the source folder
        for filename in os.listdir(src_folder):
            if filename.endswith('.npy'):  # Process only numpy files
                file_path = os.path.join(src_folder, filename)

                # Load the numpy file
                data = np.load(file_path)

                # Process the data to extract valuable sections and save
                Get_valuable_section(data, dst_folder, filename, Adq_Freq, Time_Captured_ms, Particle_Params)


# Run the main function to process all files and save the results
process_and_save_files(path_folder, Name_Folder, Final_Folder)
