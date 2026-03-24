# Mini Segmentation Projects

A compact collection of computer vision mini-projects focused on video analysis, motion monitoring, and segmentation workflows. The repository includes security-oriented detection, live object segmentation, and people segmentation examples.

## Overview

This repository demonstrates practical computer vision pipelines built with Python and OpenCV. Each project is self-contained and can be executed independently.

## Projects

| Project | Description | Main Script |
| --- | --- | --- |
| Burglar Detection | Motion-based security monitoring with visual overlays and alarm support | `Burglar detection/burglar detection.py` |
| Live Object Segmentation | Object segmentation on live video streams | `segment objects from a live video/objects segmentation.py` |
| People Segmentation | People segmentation for images and masked outputs | `segment people/pepole segmentation.py` |

## Repository Structure

```text
.
├── Burglar detection/
│   └── burglar detection.py
├── segment objects from a live video/
│   └── objects segmentation.py
├── segment people/
│   ├── pepole segmentation.py
│   ├── image/
│   └── masked images/
└── README.md
```

## Key Features

- Real-time video processing
- Motion detection with on-screen annotations
- Alarm and alert support for security scenarios
- Object and people segmentation workflows
- Saved output videos and processed results

## Requirements

- Python 3.7 or later
- OpenCV
- NumPy
- Additional packages depending on the selected script

## Setup

1. Clone the repository:

  ```bash
  git clone https://github.com/samir-m0hamed/projects.git
  cd projects
  ```

2. Install the core dependencies:

  ```bash
  pip install opencv-python numpy
  ```

3. If a project requires extra libraries, install them according to the script comments or error messages.

## How to Run

Run each project from its folder:

```bash
# Burglar Detection
python "Burglar detection/burglar detection.py"

# Live Object Segmentation
python "segment objects from a live video/objects segmentation.py"

# People Segmentation
python "segment people/pepole segmentation.py"
```

## Output

- Burglar detection saves processed videos inside `burglar_detection_results/`
- People segmentation stores generated masks and related images in the `segment people/` folders
- Each script prints progress and processing details in the terminal

## Notes

- The file name `pepole segmentation.py` is preserved as it exists in the project.
- The burglar detection script is motion-based and not Mask R-CNN based.
- Some videos may be encoded differently, so frame counts and playback behavior can vary slightly by file.

## Contributing

Contributions are welcome. If you add new projects, keep the folder structure clear and update this README accordingly.

## License

No explicit license has been provided yet. Add one if you plan to share or distribute the repository publicly.

## Author

Samir Mohamed
