# Mini Segmentation Projects with Mask R-CNN

A collection of computer vision projects demonstrating object segmentation using Mask R-CNN. These projects focus on real-time video processing for various applications including security, surveillance, and object detection.

## Projects Included

### 1. Burglar Detection
- **Description**: Motion detection system for burglar alert using computer vision.
- **Features**:
  - Real-time motion detection
  - Alarm system with sound alerts
  - Video recording with detection overlays
  - Configurable sensitivity settings
- **Files**: `Burglar detection/burglar detection.py`

### 2. Segment Objects from Live Video
- **Description**: Real-time object segmentation from live video streams.
- **Features**:
  - Live video processing
  - Object detection and segmentation
  - Mask R-CNN implementation
- **Files**: `segment objects from a live video/objects segmentation.py`

### 3. Segment People
- **Description**: People segmentation from images and videos.
- **Features**:
  - Person detection and masking
  - Image processing
  - Mask generation
- **Files**: `segment people/pepole segmentation.py`, `segment people/image/`, `segment people/masked images/`

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Mask R-CNN (if applicable)
- Other dependencies as listed in project files

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/samir-m0hamed/projects.git
   cd projects
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy
   # Install other required packages as needed
   ```

## Usage

Each project has its own Python script. Run them individually:

```bash
# Burglar Detection
python "Burglar detection/burglar detection.py"

# Object Segmentation
python "segment objects from a live video/objects segmentation.py"

# People Segmentation
python "segment people/pepole segmentation.py"
```

## Features

- **Motion Detection**: Advanced algorithms for detecting movement in video streams
- **Real-time Processing**: Optimized for live video analysis
- **Alarm System**: Integrated sound alerts for security applications
- **Video Output**: Processed videos with detection overlays
- **Configurable Parameters**: Adjustable sensitivity and thresholds

## Contributing

Feel free to contribute improvements, bug fixes, or new features.

## License

This project is open source. Please check individual files for specific licenses.

## Author

Samir Mohamed