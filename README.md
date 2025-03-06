# Rock Paper Scissors Gesture Recognition

A real-time hand gesture recognition system that detects Rock, Paper, and Scissors hand gestures using computer vision and deep learning.

## Project Overview

This project implements a CNN-based gesture recognition system capable of classifying hand gestures into Rock, Paper, or Scissors in real-time. It combines hardware (ESP32-S3 Sense) with deep learning to create an interactive and responsive system.

## Repository Structure

- **CNN_training/**: Contains the TensorFlow implementation of the EfficientNetB0 model used for training.
- **dataset/**: Image dataset used for training the model.
- **esp32_s3_sense/**: Microcontroller code for the ESP32-S3 Sense camera module.
- **stream_viewer.py**: Python script for real-time visualization of the camera stream.

## CNN Architecture

The project uses transfer learning with EfficientNetB0 as the base model:
- Pre-trained on ImageNet dataset
- Fine-tuned on custom RPS gesture dataset
- Implements progressive training: frozen base model followed by fine-tuning
- Uses regularization techniques to prevent overfitting

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- ESP32-S3 Sense development board

### Training the Model
```bash
cd CNN_training
python train.py
```

### Running the Stream Viewer
```bash
python stream_viewer.py
```

### Hardware Setup
1. Flash the ESP32-S3 Sense with the provided code
2. Connect the device to your computer via USB
3. Run the stream viewer to see real-time recognition

## Future Improvements
- Enhanced data augmentation
- Model optimization for even faster inference
- Progressive unfreezing during fine-tuning

## License
[MIT License](LICENSE)

## Acknowledgements
- TensorFlow team for EfficientNet implementation
- ESP32-S3 Sense documentation and community
