# MiniProject - Autonomous Camera Director

## Overview
This project implements an intelligent camera director that combines real-time person detection with audio processing for automatic camera focusing in lecture halls. It uses YOLOv8 for person detection and includes advanced audio features like voice activity detection, speaker classification, and direction of arrival estimation.

## Project Structure
```
MiniProject
├── src
│   ├── main.py                    # Original YOLO-only entry point
│   ├── audio_director.py          # Real-time audio processing module
│   ├── audio_camera_integration.py # Integrated audio-visual director
│   ├── auto_frame.py              # Functions for auto-framing detected persons
│   ├── detectors
│   │   └── yolov8_detector.py     # YOLOv8 model loading and detection
│   ├── utils
│   │   └── smoothing.py           # Utility functions for smoothing transitions
│   └── __init__.py                # Marks the src directory as a Python package
├── models
│   └── yolov8n.pt                 # Pre-trained YOLOv8 model file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files and directories to ignore by Git
└── README.md                      # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**
   Clone the project repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**
   Change to the project directory:
   ```
   cd MiniProject
   ```

3. **Create a Virtual Environment (Optional but Recommended)**
   Create a virtual environment to manage dependencies:
   ```
   python -m venv venv
   ```
   Activate the virtual environment:
   - For Command Prompt:
     ```
     venv\Scripts\activate
     ```
   - For PowerShell:
     ```
     .\venv\Scripts\Activate.ps1
     ```

4. **Install Dependencies**
   Install the required Python packages using:
   ```
   pip install -r requirements.txt
   ```

5. **Run the Application**
   
   **Option 1: Original YOLO-only version**
   ```
   python src/main.py
   ```
   
   **Option 2: Audio-only processing (for testing)**
   ```
   python src/audio_director.py
   ```
   
   **Option 3: Integrated audio-visual director (recommended)**
   ```
   python src/audio_camera_integration.py
   ```

## Features

### Visual Processing
- Real-time person detection using YOLOv8
- Automatic camera focusing on detected persons
- Smooth transitions and bounding box tracking
- Full-screen display mode

### Audio Processing
- **Voice Activity Detection (VAD)**: Uses Silero VAD for accurate speech detection
- **Feature Extraction**: MFCC and LPC features for speaker analysis
- **Speaker Classification**: Trainable classifier to distinguish lecturer vs audience
- **Direction of Arrival (DOA)**: Optional support for microphone arrays
- **Real-time Processing**: Low-latency audio processing in separate thread

### Integration Features
- Speech-triggered camera focus changes
- Audio-visual synchronization
- Training mode for speaker classification
- Configurable audio parameters

## Usage

### Basic Usage
- The application will open a window displaying the webcam feed
- It will automatically detect and track persons in the frame
- Audio processing runs in the background and influences camera behavior
- Press 'q' to exit the application

### Training Speaker Classification
1. Press 't' to start training mode
2. Enter label: 0 for audience, 1 for lecturer
3. Press '0' or '1' to add training samples while speaking
4. Press 's' to stop training and train the classifier
5. The system will now automatically classify speakers

### Audio Controls
- The system automatically detects speech and adjusts camera behavior
- Green bounding boxes indicate speech is active
- Red bounding boxes indicate no speech detected
- Console shows real-time audio events and feature information

## License
This project is licensed under the MIT License. See the LICENSE file for more details.