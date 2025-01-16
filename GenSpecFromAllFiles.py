import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft, cwt, ricker, hilbert
import pywt
from scipy.fftpack import fft
from librosa.feature import mfcc
import librosa
from sklearn.preprocessing import MinMaxScaler

# Base paths
base_csv_path = r"C:\InputFolder" 
base_output_path = r"C:\OutputFolder"
resolution=30
# List of classes from C1 to C10
classes = [f"C{i}" for i in range(1, 11)]


def process_cwt_spectrogram(data, output_folder, widths=np.arange(1, 31), sampling_rate=512, wavelet='morl'):
    """
    Generate Continuous Wavelet Transform (CWT) spectrograms
    """
    if data.empty:
        print("Data is empty.")
        return

    x = data.iloc[:, 0].values  # Get x-axis values
    data_cols = data.iloc[:, 1:]  # Get all columns except the first

    for column in data_cols.columns:
        print(f"Processing column: {column}")
        y = data_cols[column].values

        # Perform CWT using PyWavelets
        cwtmatr, frequencies = pywt.cwt(y, widths, wavelet)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(x, widths, np.abs(cwtmatr), cmap='jet', shading='gouraud')
        plt.axis('off')

        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved CWT image for {column} at {plot_filename}")


def process_stft_spectrogram(data, output_folder, sampling_rate=512, nperseg=256):
    """
    Generate Short-Time Fourier Transform (STFT) spectrograms
    """
    if data.empty:
        print("Data is empty.")
        return

    x = data.iloc[:, 0].values  # Get x-axis values
    data_cols = data.iloc[:, 1:]  # Get all columns except the first

    for column in data_cols.columns:
        print(f"Processing column: {column}")
        y = data_cols[column].values
        f, t, Zxx = stft(y, fs=sampling_rate, nperseg=nperseg)

        t_scaled = np.interp(t * sampling_rate, np.arange(len(y)), x)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t_scaled, f, np.abs(Zxx), shading='gouraud', cmap='jet')
        plt.axis('off')

        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved STFT image for {column} at {plot_filename}")

def process_gaf_spectrogram(data, output_folder, image_size=128):
    """
    Generate Gramian Angular Field (GAF) spectrograms
    Uses first column for data scaling
    """
    x = data.iloc[:, 0].values
    data_cols = data.iloc[:, 1:]
    #data_cols = data.iloc[:, 1]  # Get only  2nd columns
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for column in data_cols.columns:
        y = data_cols[column].values
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        # Clip values to ensure they are in the valid range for arccos
        y_scaled = np.clip(y_scaled, -1, 1)

        # Calculate GAF matrix
        phi = np.arccos(y_scaled)
        r, c = np.meshgrid(phi, phi)
        gaf = np.cos(r + c)

        plt.figure(figsize=(8, 8))
        plt.imshow(gaf, cmap='rainbow', origin='lower', extent=[x[0], x[-1], x[0], x[-1]])
        plt.axis('off')

        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved GAF image for {column}")

def process_hht_spectrogram(data, output_folder, sampling_rate=512):
    """
    Generate Hilbert-Huang Transform (HHT) spectrograms
    Uses first column as x-axis values
    """
    if data.empty:
        print("Data is empty.")
        return

    x = data.iloc[:, 0].values  # First column: time or x-axis
    data_cols = data.iloc[:, 1:]  # All other columns: signal data

    for column in data_cols.columns:
        print(f"Processing column: {column}")
        y = data_cols[column].values

        # Perform Hilbert transform
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_rate

        # Prepare grid and amplitude envelope for visualization
        x_grid, y_grid = np.meshgrid(x[:-1], np.arange(len(amplitude_envelope[:-1])))

        # Make C match the shape of x_grid and y_grid
        C = np.tile(amplitude_envelope[:-1], (len(x[:-1]), 1)).T

        plt.figure(figsize=(10, 6))

        # Use shading='auto' to align dimensions
        plt.pcolormesh(x_grid, y_grid, C, cmap='jet', shading='auto')
        plt.axis('off')

        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved HHT image for {column} at {plot_filename}")




def process_dwt_spectrogram(data, output_folder, wavelet='db4', level=5):
    """
    Generate Discrete Wavelet Transform (DWT) spectrograms
    Uses first column for x-axis scaling
    """
    x = data.iloc[:, 0].values
    data_cols = data.iloc[:, 1:]
    #data_cols = data.iloc[:, 1]  # Get only  2nd columns

    for column in data_cols.columns:
        y = data_cols[column].values

        # Perform DWT
        coeffs = pywt.wavedec(y, wavelet, level=level)

        # Plot coefficients with scaled x-axis
        plt.figure(figsize=(10, 6))
        for i, coeff in enumerate(coeffs):
            plt.subplot(level + 1, 1, i + 1)
            x_scaled = np.linspace(x[0], x[-1], len(coeff))
            plt.plot(x_scaled, coeff)
            plt.axis('off')

        plt.tight_layout()
        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved DWT image for {column}")


def process_mfcc_spectrogram(data, output_folder, sampling_rate=512, n_mfcc=13):
    """
    Generate Mel-frequency cepstral coefficients (MFCC) spectrograms
    Uses first column for time scaling
    """
    x = data.iloc[:, 0].values
    data_cols = data.iloc[:, 1:]
    #data_cols = data.iloc[:, 1]  # Get only  2nd columns

    for column in data_cols.columns:
        y = data_cols[column].values

        # Calculate MFCC
        mfccs = mfcc(y=y, sr=sampling_rate, n_mfcc=n_mfcc)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(np.linspace(x[0], x[-1], mfccs.shape[1]),
                       np.arange(mfccs.shape[0]),
                       mfccs, cmap='jet')
        plt.axis('off')

        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved MFCC image for {column}")


def process_rp_spectrogram(data, output_folder, embedding_dimension=10, time_delay=2):
    """
    Generate Recurrence Plot (RP) spectrograms
    Uses first column for plot scaling
    """
    x = data.iloc[:, 0].values
    data_cols = data.iloc[:, 1:]
    #data_cols = data.iloc[:, 1]  # Get only  2nd columns

    for column in data_cols.columns:
        y = data_cols[column].values

        # Create time-delay embedding
        N = len(y) - (embedding_dimension - 1) * time_delay
        Y = np.zeros((N, embedding_dimension))
        for i in range(embedding_dimension):
            Y[:, i] = y[i * time_delay:i * time_delay + N]

        # Calculate distance matrix
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = np.sqrt(np.sum((Y[i, :] - Y[j, :]) ** 2))

        plt.figure(figsize=(8, 8))
        plt.imshow(D, cmap='binary', origin='lower',
                   extent=[x[0], x[-1], x[0], x[-1]])
        plt.axis('off')

        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved RP image for {column}")


def process_spectral_entropy(data, output_folder, window_size=256, overlap=128):
    """
    Generate Spectral Entropy visualizations
    Uses first column as x-axis values
    """
    x = data.iloc[:, 0].values
    data_cols = data.iloc[:, 1:]
    #data_cols = data.iloc[:, 1]  # Get only  2nd columns

    for column in data_cols.columns:
        y = data_cols[column].values

        # Calculate spectral entropy for each window
        entropy_values = []
        x_values = []
        for i in range(0, len(y) - window_size, overlap):
            window = y[i:i + window_size]
            spectrum = np.abs(fft(window)) ** 2
            spectrum_normalized = spectrum / np.sum(spectrum)
            entropy = -np.sum(spectrum_normalized * np.log2(spectrum_normalized + 1e-10))
            entropy_values.append(entropy)
            x_values.append(x[i + window_size // 2])

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, entropy_values)
        plt.axis('off')

        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved Spectral Entropy image for {column}")


def process_tf_embedding(data, output_folder, sampling_rate=512):
    """
    Generate Time-Frequency Embedding visualizations using multiple representations
    Uses first column as x-axis values
    """
    x = data.iloc[:, 0].values
    data_cols = data.iloc[:, 1:]
    #data_cols = data.iloc[:, 1]  # Get only  2nd columns

    for column in data_cols.columns:
        y = data_cols[column].values

        # Calculate different time-frequency representations
        f, t, Zxx = stft(y, fs=sampling_rate)
        S = librosa.feature.melspectrogram(y=y, sr=sampling_rate)

        # Scale time axes to match x values
        t_scaled = np.interp(t * sampling_rate, np.arange(len(y)), x)
        mel_x = np.linspace(x[0], x[-1], S.shape[1])

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.pcolormesh(t_scaled, f, np.abs(Zxx), shading='gouraud', cmap='jet')
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.pcolormesh(mel_x, np.arange(S.shape[0]),
                       librosa.power_to_db(S, ref=np.max), cmap='jet')
        plt.axis('off')

        plt.tight_layout()
        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename, format='png', dpi=resolution, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved TF Embedding image for {column}")

def main(base_csv_path, base_output_path, classes):
    for class_name in classes:
        csv_file_path = os.path.join(base_csv_path, f"{class_name}.csv")
        output_folder = os.path.join(base_output_path, class_name)
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing class: {class_name}")
        print(f"Reading file: {csv_file_path}")

        try:
            data = pd.read_csv(csv_file_path)

            if data.empty:
                print(f"File {csv_file_path} is empty.")
                continue

            # Process functions (uncomment as needed)
            #process_cwt_spectrogram(data, output_folder)
            #process_gaf_spectrogram(data, output_folder)
            #process_stft_spectrogram(data, output_folder)
            #process_hht_spectrogram(data, output_folder)
            #process_dwt_spectrogram(data, output_folder)
            #process_mfcc_spectrogram(data, output_folder)
            #process_rp_spectrogram(data, output_folder)
            #process_spectral_entropy(data, output_folder)
            process_tf_embedding(data, output_folder)
            # Add other processing functions if needed

        except Exception as e:
            print(f"Error processing {class_name}: {str(e)}")


# Run the script
if __name__ == "__main__":
    main(base_csv_path, base_output_path, classes)
