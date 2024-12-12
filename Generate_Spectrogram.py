import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet

# Load the CSV file
csv_file_path = r"C:\BULabAssets\BULabProjects\BiomaterialData\dataset\Alginate\Cropped\combined_filtered_data_Alg 1% – Ca 0_2.csv" # Replace with your actual CSV file path
data = pd.read_csv(csv_file_path)
# Folder to save the plots
output_folder = r"C:\BULabAssets\BULabProjects\BiomaterialData\dataset\Alginate\Spectrogram\30pix\C2" # Replace with your desired folder name
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Exclude the first and last column
# data = data.iloc[:, 1:-1]
# Exclude the first column only
data = data.iloc[:, 1:]

# Sampling rate (adjust according to your data)
sampling_rate = 10048  # Example: 2048 Hz

# Loop through each column and save its scalogram as a plot
for column in data.columns:
    y = data[column].values  # Get the signal values of the column
    x = np.arange(len(y)) / sampling_rate  # Generate the time axis based on the sampling rate

    # Perform Continuous Wavelet Transform (CWT)
    mother = wavelet.Morlet(6)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1 / sampling_rate, wavelet=mother)

    # Create a plot for the current column
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(wave), extent=[x[0], x[-1], freqs[-1], freqs[0]], cmap='jet', aspect='auto')

    # Remove the entire axis
    plt.axis('off')

    # Save the plot with tight layout and without white borders
    plot_filename = os.path.join(output_folder, f'{column}.png')
    plt.savefig(plot_filename,
                format='png',
                dpi=30,
                bbox_inches='tight',  # Trim the whitespace around the figure
                pad_inches=0)  # Remove padding completely

    # Close the plot to free up memory
    plt.close()
    print(f"Saved CWT image for {column} as {plot_filename}")

print(f"Scalograms saved to folder: {output_folder}")
