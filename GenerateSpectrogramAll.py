import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet
from typing import List, Union, Optional
import logging

class CWTSpecrogramProcessor:
    def __init__(self, 
                 base_csv_path: str, 
                 base_output_path: str, 
                 sampling_rate: float = 10048,
                 wavelet_type: str = 'morlet',
                 wavelet_parameter: float = 6.0,
                 log_level: str = 'INFO'):
        """
        Initialize CWT Spectrogram Processor
        
        Parameters:
        - base_csv_path: Input directory containing CSV files
        - base_output_path: Output directory for spectrogram images
        - sampling_rate: Signal sampling rate
        - wavelet_type: Type of wavelet to use
        - wavelet_parameter: Parameter for wavelet transformation
        - log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Path and processing configurations
        self.base_csv_path = base_csv_path
        self.base_output_path = base_output_path
        self.sampling_rate = sampling_rate
        self.wavelet_type = wavelet_type
        self.wavelet_parameter = wavelet_parameter

        # Validate input paths
        self._validate_paths()

    def _validate_paths(self):
        """Validate input and output paths"""
        if not os.path.exists(self.base_csv_path):
            raise ValueError(f"Input CSV path does not exist: {self.base_csv_path}")
        
        os.makedirs(self.base_output_path, exist_ok=True)

    def _select_wavelet(self) -> callable:
        """
        Select and return appropriate wavelet based on configuration
        
        Supports multiple wavelet types with extensibility
        """
        wavelet_map = {
            'morlet': lambda: wavelet.Morlet(self.wavelet_parameter),
            'paul': lambda: wavelet.Paul(self.wavelet_parameter),
            'mexican_hat': lambda: wavelet.Mexican_hat()
        }
        
        if self.wavelet_type.lower() not in wavelet_map:
            self.logger.warning(f"Unsupported wavelet type. Defaulting to Morlet.")
            return wavelet_map['morlet']
        
        return wavelet_map[self.wavelet_type.lower()]

    def process_spectrogram(self, 
                             data: pd.DataFrame, 
                             output_folder: str,
                             columns: Optional[List[str]] = None):
        """
        Generate spectrograms for specified columns
        
        Parameters:
        - data: Input DataFrame
        - output_folder: Destination for spectrogram images
        - columns: Optional list of specific columns to process
        """
        # Use all columns if not specified
        if columns is None:
            columns = data.columns[1:]  # Skip first column
        
        # Select wavelet
        wavelet_func = self._select_wavelet()
        
        for column in columns:
            try:
                # Extract signal
                y = data[column].values
                x = np.arange(len(y)) / self.sampling_rate

                # Perform CWT
                wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
                    y, 1 / self.sampling_rate, wavelet=wavelet_func()
                )

                # Plot spectrogram
                plt.figure(figsize=(12, 8))
                plt.imshow(
                    np.abs(wave), 
                    extent=[x[0], x[-1], freqs[-1], freqs[0]], 
                    cmap='viridis', 
                    aspect='auto'
                )
                plt.colorbar(label='Wavelet Coefficient Magnitude')
                plt.title(f'CWT Spectrogram: {column}')
                plt.xlabel('Time')
                plt.ylabel('Frequency')

                # Save with higher resolution
                plot_filename = os.path.join(output_folder, f'{column}_spectrogram.png')
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()

                self.logger.info(f"Generated spectrogram for {column}")

            except Exception as e:
                self.logger.error(f"Error processing {column}: {e}")

    def batch_process(self, 
                      classes: Union[List[str], str] = None, 
                      file_pattern: str = "{class_name}.csv"):
        """
        Batch process multiple CSV files
        
        Parameters:
        - classes: List of class names or pattern
        - file_pattern: Filename pattern
        """
        # Generate classes if not provided
        if classes is None:
            classes = [f"C{i}" for i in range(1, 10)]
        elif isinstance(classes, str):
            classes = [classes]

        for class_name in classes:
            try:
                # Construct paths
                csv_path = os.path.join(
                    self.base_csv_path, 
                    file_pattern.format(class_name=class_name)
                )
                output_folder = os.path.join(self.base_output_path, class_name)
                os.makedirs(output_folder, exist_ok=True)

                # Read and process CSV
                if os.path.exists(csv_path):
                    data = pd.read_csv(csv_path)
                    self.logger.info(f"Processing {class_name}")
                    self.process_spectrogram(data, output_folder)
                else:
                    self.logger.warning(f"CSV not found: {csv_path}")

            except Exception as e:
                self.logger.error(f"Batch processing error for {class_name}: {e}")

def main():
    processor = CWTSpecrogramProcessor(
        base_csv_path=r"C:\CSVs\Seperate",
        base_output_path=r"C:\Spectrogram\Improved",
        sampling_rate=10048,
        wavelet_type='morlet',
        log_level='DEBUG'
    )
    processor.batch_process()

if __name__ == "__main__":
    main()
