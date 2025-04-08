# AI-Powered Real-Time Smoking Detection System

This system uses computer vision and AI to detect smoking behavior in real-time based on the relative position of hands and face.

## Features

- Real-time hand and face detection using OpenCV and cvzone
- Smoking gesture recognition by tracking hand and nose positions
- Audio alerts using text-to-speech when smoking is detected
- Visual warning overlays on the video feed
- Multithreaded design for smooth performance

## Requirements

- Python 3.7+
- Webcam

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the smoking detection system:

```bash
python smoking_detection.py
```

- Press 'q' to quit the application
- The system will automatically detect faces and hands
- If a hand is detected near the mouth region, it will trigger smoking alerts
- Audio and visual warnings will be displayed when smoking is detected

## Customization

You can adjust detection sensitivity by modifying the following parameter in the code:

- `mouth_region_threshold`: Distance threshold for detecting hand proximity to face (default: 100 pixels)

## Note

This system works best with good lighting conditions and when the face is clearly visible to the camera. 