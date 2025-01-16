# Bicubic Interpolation and Binary Thresholding
This Python script uses the PIL (Python Imaging Library) to perform image processing operations including bicubic interpolation and binary thresholding on a specified input image. Here's a summary of its functionality:

### Import Libraries
- Imports necessary libraries, including `Image` from PIL.

### Define Functions
- **`bicubic_interpolation(image, target_size)`**: Resizes the input image to the target size using bicubic interpolation.
- **`convert_to_binary(image, threshold)`**: Converts the input image to grayscale and applies binary thresholding based on a specified threshold value.

### Main Function
- **Load Image**: Loads the input image from a specified path (`input_image_path`).
- **Resize Image**: Resizes the original image to a fixed target size (28x28) using bicubic interpolation.
- **Display Resized Image**: Displays the resized original image.
- **Bicubic Interpolation**: Applies bicubic interpolation to the resized image. The scale factor can be adjusted, but it's set to 1 in this example.
- **Display Bicubic Image**: Displays the bicubic interpolated image.
- **Convert to Binary Image**: Converts the bicubic interpolated image to a binary image using a specified threshold value (200).
- **Display Binary Image**: Displays the binary image.
- **Save Images**: Saves the resized image, bicubic interpolated image, and binary image to disk with specified quality settings.

### Execution
- The `main()` function is called if the script is run directly.

This script demonstrates resizing an image, applying interpolation, converting to binary, displaying the processed images, and saving the results to disk.


# Continuous Wavelet Transform (CWT) Scalogram Generator

## Code Breakdown and Implementation Steps

### 1. Import Required Libraries
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet
```
- `os`: For file and directory operations
- `numpy`: Numerical computing and array manipulation
- `pandas`: Data manipulation and CSV file handling
- `matplotlib.pyplot`: Plotting and visualization
- `pycwt`: Continuous Wavelet Transform library

### 2. Load Data and Prepare Output Directory
```python
# Load the CSV file
csv_file_path = r"C:\data.csv"
data = pd.read_csv(csv_file_path)

# Create output folder for scalogram images
output_folder = r"C:\Spectrogram"
os.makedirs(output_folder, exist_ok=True)
```
- Reads CSV file into a pandas DataFrame
- Creates output directory if it doesn't exist
- `exist_ok=True` prevents errors if directory already exists

### 3. Data Preprocessing
```python
# Exclude the first column only
data = data.iloc[:, 1:]
```
- Removes the first column from the DataFrame
- `iloc[:, 1:]` selects all rows and starts from the second column

### 4. Set Sampling Rate
```python
sampling_rate = 10048  # Hz
```
- Defines the sampling frequency of the signal
- Critical for accurate time-frequency representation
- Should match the actual data collection rate

### 5. Continuous Wavelet Transform and Visualization Loop
```python
for column in data.columns:
    # Extract signal values
    y = data[column].values
    
    # Generate time axis
    x = np.arange(len(y)) / sampling_rate
    
    # Perform Continuous Wavelet Transform
    mother = wavelet.Morlet(6)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1 / sampling_rate, wavelet=mother)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(wave), 
               extent=[x[0], x[-1], freqs[-1], freqs[0]], 
               cmap='jet', 
               aspect='auto')
    
    # Remove axis for clean visualization
    plt.axis('off')
    
    # Save the scalogram
    plot_filename = os.path.join(output_folder, f'{column}.png')
    plt.savefig(plot_filename,
                format='png',
                dpi=30,
                bbox_inches='tight',
                pad_inches=0)
    
    # Clean up
    plt.close()
    print(f"Saved CWT image for {column} as {plot_filename}")
```

#### Key Steps in the Loop:
1. **Signal Extraction**
   - Extracts values for each column as a time series
   - Generates corresponding time axis based on sampling rate

2. **Continuous Wavelet Transform**
   - Uses Morlet wavelet (parameter 6) for analysis
   - `wavelet.cwt()` performs the transform
   - Converts time-domain signal to time-frequency representation

3. **Visualization**
   - Creates a figure for each column's scalogram
   - Uses `imshow()` to display the wavelet transform magnitude
   - `extent` parameter maps the image to actual time and frequency axes
   - `cmap='jet'` provides a color gradient representation
   - `aspect='auto'` adjusts the plot's aspect ratio

4. **Image Saving**
   - Saves each scalogram as a PNG
   - `dpi=30` for lower resolution (adjust as needed)
   - `bbox_inches='tight'` removes whitespace
   - `pad_inches=0` eliminates padding

### 6. Completion Notification
```python
print(f"Scalograms saved to folder: {output_folder}")
```
- Confirms the process is complete
- Indicates the folder where images are saved

## Scalogram Interpretation
- Bright colors (red/yellow) indicate high energy
- Dark colors (blue/green) indicate low energy
- Vertical axis represents frequency
- Horizontal axis represents time
- Helps visualize signal characteristics across different time and frequency scales

## Potential Customizations
- Adjust `sampling_rate` to match your data
- Change `cmap` for different color schemes
- Modify `dpi` for image resolution
- Adjust Morlet wavelet parameter for different analysis needs

**Understanding the Code: A Step-by-Step Guide**

**1. Import Necessary Libraries**
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet
```
* **os**: Used for file system operations, like creating directories.
* **numpy**: Provides efficient numerical operations on arrays.
* **pandas**: A powerful data analysis and manipulation tool for working with tabular data.
* **matplotlib.pyplot**: A plotting library for creating visualizations.
* **pycwt**: A library for performing Continuous Wavelet Transform (CWT) analysis.

**2. Load the CSV Data**
```python
csv_file_path = r"C:\data.csv"  # Replace with your actual CSV file path
data = pd.read_csv(csv_file_path)
```
* Reads the specified CSV file into a pandas DataFrame.

**3. Prepare the Data**
```python
# Exclude the first column only
data = data.iloc[:, 1:]
```
* Removes the first column from the DataFrame, as it might contain unnecessary information like timestamps or labels.

**4. Set the Sampling Rate**
```python
sampling_rate = 10048  # Example: 2048 Hz
```
* Defines the sampling rate of the data, which is essential for accurate time-frequency analysis.

**5. Iterate Over Columns and Perform CWT**
```python
for column in data.columns:
    y = data[column].values  # Get the signal values of the column
    x = np.arange(len(y)) / sampling_rate  # Generate the time axis based on the sampling rate
    mother = wavelet.Morlet(6)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1 / sampling_rate, wavelet=mother)
```
* **Iterates over each column:** Processes each column individually.
* **Extracts signal values:** Retrieves the numerical values from the current column.
* **Generates time axis:** Creates a time axis based on the sampling rate.
* **Performs CWT:** Applies the Continuous Wavelet Transform using a Morlet wavelet with a wavelet parameter of 6.

**6. Visualize and Save Scalograms**
```python
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(wave), extent=[x[0], x[-1], freqs[-1], freqs[0]], cmap='jet', aspect='auto')
plt.axis('off')
plot_filename = os.path.join(output_folder, f'{column}.png')
plt.savefig(plot_filename, format='png', dpi=30, bbox_inches='tight', pad_inches=0)
plt.close()
```
* **Creates a plot:** Initializes a figure and plots the absolute value of the CWT coefficients as an image.
* **Removes axis:** Turns off the axis labels for a cleaner visualization.
* **Saves the plot:** Saves the plot as a PNG image in the specified output folder.
* **Closes the plot:** Releases memory used by the plot.

**Key Points:**
- **CWT:** A powerful tool for analyzing time-frequency properties of signals.
- **Morlet wavelet:** A commonly used wavelet for analyzing oscillatory signals.
- **Scalogram:** A visual representation of the time-frequency energy distribution of a signal.
- **Figure size and axis removal:** Customize the plot appearance for better visualization.
- **Saving plots:** Saves the plots as PNG images with high-quality settings.

By following these steps, the code effectively processes each column of the CSV data, performs CWT, visualizes the results as scalograms, and saves them as individual PNG images.

### Purpose of the Script ""

The script is designed to generate various types of time-frequency and signal transformation visualizations for input data stored in CSV files. It includes methods for generating **Continuous Wavelet Transform (CWT)**, **Short-Time Fourier Transform (STFT)**, **Gramian Angular Field (GAF)**, **Hilbert-Huang Transform (HHT)**, **Discrete Wavelet Transform (DWT)**, **Mel-Frequency Cepstral Coefficients (MFCC)**, **Recurrence Plots (RP)**, **Spectral Entropy**, and **Time-Frequency Embeddings**. These visualizations can be used for feature extraction in machine learning, deep learning, or signal analysis tasks, particularly in domains like biomedical signal processing, speech recognition, and pattern recognition.

---

### Step-by-Step Implementation

#### 1. **Setup and Imports**
The script imports essential Python libraries for data manipulation, signal processing, and visualization:
- `os` for file and directory operations.
- `numpy` and `pandas` for numerical computations and data handling.
- `matplotlib` for creating plots.
- Signal processing libraries such as `scipy.signal`, `pywt`, `librosa`, and `sklearn.preprocessing`.

#### 2. **Define Base Paths and Parameters**
- `base_csv_path`: Directory containing input CSV files (`C1.csv` to `C10.csv`).
- `base_output_path`: Directory where generated images will be saved.
- `classes`: List of class names, e.g., `C1`, `C2`, ..., `C10`.
- `resolution`: Controls the DPI (dots per inch) for saving generated plots.

#### 3. **Define Individual Processing Functions**

Each processing function takes an input DataFrame, processes the signal data, and generates visualizations for each column (except the first, which is used as the x-axis).

##### a. **Continuous Wavelet Transform (CWT) Spectrogram**
- Uses PyWavelets (`pywt.cwt`) to compute the CWT for a range of widths.
- Visualizes the resulting coefficients as a spectrogram using `plt.pcolormesh`.

##### b. **Short-Time Fourier Transform (STFT) Spectrogram**
- Computes the STFT using `scipy.signal.stft`.
- Produces a time-frequency spectrogram by plotting the magnitude of the STFT coefficients.

##### c. **Gramian Angular Field (GAF)**
- Scales the signal into a range [-1, 1].
- Constructs a GAF matrix based on trigonometric transformations.
- Visualizes the GAF matrix as a 2D image.

##### d. **Hilbert-Huang Transform (HHT)**
- Computes the analytic signal using the Hilbert Transform (`scipy.signal.hilbert`).
- Visualizes the amplitude envelope over time using `plt.pcolormesh`.

##### e. **Discrete Wavelet Transform (DWT)**
- Decomposes the signal into wavelet coefficients using `pywt.wavedec`.
- Plots the coefficients at different decomposition levels.

##### f. **Mel-Frequency Cepstral Coefficients (MFCC)**
- Computes MFCC features using `librosa.feature.mfcc`.
- Visualizes the MFCC coefficients as a spectrogram.

##### g. **Recurrence Plot (RP)**
- Embeds the signal in a time-delay space.
- Computes the Euclidean distance matrix for the embedding.
- Visualizes the distance matrix as a recurrence plot.

##### h. **Spectral Entropy**
- Computes the spectral entropy for sliding windows of the signal.
- Plots the entropy values over time.

##### i. **Time-Frequency Embedding**
- Combines STFT and Mel-spectrogram visualizations in a single figure.
- Scales the time axes to match the input signal.

#### 4. **Process Each CSV File**
- Reads each file corresponding to classes `C1` through `C10`.
- Validates if the file exists and is non-empty.
- Executes one or more of the defined processing functions (uncomment lines in the `main` function to use specific methods).

#### 5. **Save the Results**
- Each function saves the generated visualization as a `.png` file in a subdirectory under `base_output_path`, named after the class.

#### 6. **Main Function**
The `main` function orchestrates the workflow:
- Iterates through all classes (`C1` to `C10`).
- Reads the corresponding CSV file into a Pandas DataFrame.
- Creates output directories for saving the results.
- Calls the desired processing functions.

#### 7. **Run the Script**
The script executes the `main` function when run directly:
```python
if __name__ == "__main__":
    main(base_csv_path, base_output_path, classes)
```

---

### Usage Instructions
1. Place your input CSV files in the directory specified by `base_csv_path`.
2. Ensure each CSV file has:
   - The first column as the x-axis (e.g., time).
   - Remaining columns as signal data.
3. Modify the `main` function to uncomment the desired processing functions.
4. Run the script. Generated spectrograms will be saved in the respective class subdirectories under `base_output_path`.

---

This modular and customizable script simplifies the generation of various signal transformations and visualizations, making it suitable for exploratory data analysis and preprocessing in signal-related tasks.


<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>
