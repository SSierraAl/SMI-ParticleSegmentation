
#NOT usefull for now
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, decimate

# -----------------------
# Config
# -----------------------
#FOLDER ="C:/Users/ssierra/Downloads/OFI_Flow_Citometry_Repo/OFI-Flow-Citometry/GlobalGUI/Particles_Data/C1_HF_5_10_4um_doublet-copia" 
FOLDER= 'C:/Users/ssierra/Documents/PHD/Irati/Noise/'

fs = 2000000  # 2 MHz original sampling rate
LOWCUT = 8000      # Hz (band-pass)
HIGHCUT = 35000     # Hz
ORDER = 4

# NEW: decimation factor (1 = no decimation)
DECIM = 4

# Spectrogram params (exactly as requested)
nperseg = 128
noverlap = 64  # overlap (samples)

def FFT_calc(datos, samplefreq):
    """
    Calculate the FFT (Fast Fourier Transform) of the input data.
    """
    n = len(datos)
    datos = np.asarray(datos, dtype=np.float64)
    fft_result = np.fft.rfft(datos)  # Compute FFT for real input
    freq_fft = np.fft.rfftfreq(n, 1 / samplefreq)  # Hz
    amplitude = np.abs(fft_result)  # magnitude
    phase = np.angle(fft_result)
    return amplitude, freq_fft, phase

# Bandpass filter (+ optional decimation)
def butter_bandpass_filter(data, lowcut, highcut, order, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)

    # apply decimation if DECIM > 1
    if DECIM > 1:
        y = decimate(y, DECIM)
        fs_eff = fs / DECIM
    else:
        fs_eff = fs

    #np.save('filtered_signal.npy', y)  # Save filtered signal for inspection
    return y, fs_eff


# -----------------------
def extract_features_with_params(signal, nperseg, noverlap, fs_hz):
    frame_step = int(nperseg - noverlap)
    spectrogram = tf.signal.stft(
        tf.convert_to_tensor(signal, dtype=tf.float32),
        frame_length=int(nperseg),
        frame_step=frame_step,
        fft_length=int(nperseg),
        pad_end=True,
    )
    Sxx = tf.abs(spectrogram)  # (frames, freq_bins)
    num_bins = Sxx.shape[1]

    # convert desired Hz-ROI to bins using the CURRENT fs (after decimation)
    frequency_per_bin = fs_hz / (2 * num_bins)  # Hz per bin up to Nyquist
    fmin_hz, fmax_hz = 7000, 38000
    # cap to Nyquist to avoid empty slices after decimation
    fmax_hz = min(fmax_hz, 0.5 * fs_hz - 1.0)

    min_bin = max(0, int(fmin_hz / frequency_per_bin))
    max_bin = min(num_bins, int(fmax_hz / frequency_per_bin))

    if max_bin <= min_bin:   # fallback if ROI collapses
        min_bin, max_bin = 0, num_bins

    Sxx = Sxx[:, min_bin:max_bin]
    return Sxx.numpy()  # (frames, freq_bins_in_ROI)

# -----------------------
# Main: process and plot each file individually
# -----------------------
def process_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    npy_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".npy")]
    if not npy_files:
        print("No .npy files found.")
        return

    for filename in npy_files:
        fpath = os.path.join(folder_path, filename)
        print(f"\nProcessing: {fpath}")

        # Load
        sig = np.load(fpath)
        if sig.ndim > 1:
            sig = sig[:, 0]  # take first channel if multi-channel

        # Filter (+ decimate) -> returns effective fs after decimation
        sig_filt, fs_eff = butter_bandpass_filter(sig, LOWCUT, HIGHCUT, ORDER, fs)

        # FFT -> uses fs_eff
        amp, freq_fft, _ = FFT_calc(sig_filt, fs_eff)

        # Spectrogram -> uses fs_eff for bin mapping
        Sxx = extract_features_with_params(sig_filt, nperseg, noverlap, fs_eff)

        # -------- Plot: 4-row subplot (time, FFT, spectrogram, binary spectrogram) --------
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)

        # 1) Time-domain (post-filter)
        axes[0].plot(sig_filt, linewidth=0.8)
        axes[0].set_title(f'Filtered temporal signal — {filename}')
        axes[0].set_xlabel('Samples')   # keep simple (not scaled to time units)
        axes[0].set_ylabel('Amplitude [a.u.]')
        axes[0].grid(True)

        # 2) FFT magnitude
        axes[1].plot(freq_fft, amp, linewidth=0.8)
        axes[1].set_title('FFT magnitude (post-filter)')
        axes[1].set_xlabel('Frequency [Hz]')
        axes[1].set_ylabel('Amplitude [a.u.]')
        axes[1].set_xlim(LOWCUT, HIGHCUT)
        axes[1].grid(True)

        # 3) Spectrogram (same imshow style as before)
        im = axes[2].imshow(
            Sxx.T,                 # (freq x time)
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='none'
        )
        axes[2].set_title(f'nperseg={nperseg}, noverlap={noverlap} — Spectrogram (bins)')
        axes[2].set_xlabel('Time [bins]')       # bins (unchanged)
        axes[2].set_ylabel('Frequency [bins]')  # bins (cropped ROI)
        cbar = fig.colorbar(im, ax=axes[2])
        cbar.set_label('Amplitude [a.u.]')




        # 4) Binary spectrogram by global mean threshold (with >=4 consecutive rule)
        print(Sxx.mean())
        print(Sxx.max())

        # Column means (time axis = columns in Sxx)
        col_means = Sxx.mean(axis=1)      # shape: (n_cols,)
        global_mean = Sxx.max() / 10.0    # your chosen threshold

        # Initial boolean mask: columns above threshold
        mask = col_means > global_mean    # True = candidate black column

        # Enforce runs of >= 4 consecutive Trues
        n_cols = mask.size
        mask_runs = np.zeros_like(mask, dtype=bool)
        if n_cols > 0:
            i = 0
            while i < n_cols:
                if mask[i]:
                    j = i
                    while j < n_cols and mask[j]:
                        j += 1
                    # [i, j) is a run of True; keep only if length >= 4
                    if (j - i) >= 4:
                        mask_runs[i:j] = True
                    i = j
                else:
                    i += 1

        # Build binary image: default white (255), black (0) for kept columns
        binary_Sxx = np.ones_like(Sxx, dtype=np.uint8) * 255
        if mask_runs.any():
            binary_Sxx[mask_runs, :] = 0

        # If *all* columns are black after the rule, show all white instead
        if mask_runs.all() and n_cols > 0:
            binary_Sxx[:] = 255

        im2 = axes[3].imshow(
            binary_Sxx.T,
            aspect='auto',
            origin='lower',
            cmap='gray',
            interpolation='none'
        )
        axes[3].set_title("Binary spectrogram (≥4-consecutive columns rule)")
        axes[3].set_xlabel("Time [bins]")
        axes[3].set_ylabel("Frequency [bins]")
        axes[3].grid(True)










        plt.show()

if __name__ == "__main__":
    process_folder(FOLDER)
