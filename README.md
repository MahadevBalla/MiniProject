# MiniProject

## Overview
This project implements a real-time person detection and tracking application using the YOLOv8 model. It captures video from the webcam, detects persons in the frame, and auto-frames them with a smooth transition effect.

## Project Structure
```
MiniProject
├── src
│   ├── main.py               # Entry point of the application
│   ├── auto_frame.py         # Functions for auto-framing detected persons
│   ├── detectors
│   │   └── yolov8_detector.py # YOLOv8 model loading and detection
│   ├── utils
│   │   └── smoothing.py       # Utility functions for smoothing transitions
│   └── __init__.py           # Marks the src directory as a Python package
├── models
│   └── yolov8n.pt            # Pre-trained YOLOv8 model file
├── requirements.txt          # Python dependencies
├── .gitignore                # Files and directories to ignore by Git
└── README.md                 # Documentation for the project
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
   Start the application by running the main script:
   ```
   python src/main.py
   ```

## Usage
- The application will open a window displaying the webcam feed.
- It will automatically detect and track persons in the frame.
- Press 'q' to exit the application.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.