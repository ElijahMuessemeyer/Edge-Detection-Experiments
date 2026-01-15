# Edge Detection Experiments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

Comprehensive edge detection experiments comparing different algorithms and their behavior under various conditions.

## Features

- **Multiple Algorithms**: Canny, Sobel, and Laplacian edge detection
- **Synthetic Image Generation**: Configurable test images
- **Comparative Analysis**: Side-by-side algorithm comparison
- **Noise Testing**: Algorithm performance under noisy conditions

## Files

- `edge_detection_comprehensive.py` - Main comprehensive experiment with detailed analysis
- `edge_detection_experiment.py` - Experimental implementations
- `simple_edge_detection.py` - Basic edge detection examples
- `results/` - Generated experiment result images

## Usage

```bash
python edge_detection_comprehensive.py
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib

## Results

The `results/` directory contains generated visualizations including:
- Clean image edge detection
- Noisy image comparison
- Low contrast analysis
- Threshold variation experiments
