# Automatic Number Plate Recognition (ANPR)

## Overview

This project implements an Automatic Number Plate Recognition (ANPR) system using Python and YOLOv5. It detects vehicles and recognizes license plates from images or video streams with high accuracy. The project includes real-time processing capabilities and uses Optical Character Recognition (OCR) to extract text from detected license plates.

## Features

- **Vehicle Detection**: Uses YOLOv5 to detect vehicles in images or video.
- **License Plate Recognition**: Extracts and reads license plate numbers using OCR.
- **Real-Time Processing**: Processes video streams in real-time for license plate recognition.
- **Custom Trained Model**: The license plate detection model is trained on a custom dataset to optimize recognition.

## Dataset

The license plate detection model was trained using the [License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) from Roboflow. This dataset contains images of various vehicle license plates, which were used to fine-tune the detection model for better accuracy.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/KFMBB/Automatic-number-plate-recognition.git
cd Automatic-number-plate-recognition
pip install -r requirements.txt
