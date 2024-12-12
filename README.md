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

<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>
