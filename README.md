# BIB Detection

A web-based application for detecting and recognizing BIB numbers in marathon/running event images using computer vision techniques.

## Project Overview

This project implements a BIB number detection system that can identify and extract runner identification numbers from images taken during running events. The system uses computer vision algorithms to locate, segment, and recognize the numbers on runners' bibs.

## Features

- Upload images containing runners with BIB numbers
- Automatic detection and recognition of BIB numbers
- Display of detection results with bounding boxes
- Extraction of BIB number data

## Technologies Used

- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Computer Vision: OpenCV, TensorFlow/PyTorch
- OCR: Tesseract

## Prerequisites

Before running this application, make sure you have the following installed:
- Python 3.8 or higher
- Node.js and npm (if using a JavaScript build system)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bib-detection.git
cd bib-detection
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```