# Vehicle Detection and Recognition with OpenVINO™

A streamlined Python implementation for detecting vehicles in images and videos, and recognizing their attributes (color and type) using OpenVINO™ pre-trained models.

## Features

- 🚗 **Vehicle Detection**: Detect vehicles in images and video streams
- 🎨 **Attribute Recognition**: Identify vehicle color (White, Gray, Yellow, Red, Green, Blue, Black) and type (Car, Bus, Truck, Van)
- 🎥 **Video Processing**: Process video files with frame skipping for efficiency
- 💾 **Auto-save Detections**: Automatically save frames containing detected vehicles
- ⚡ **Optimized Performance**: Built on OpenVINO for fast inference on CPU/GPU
- 🔧 **Configurable**: Adjustable confidence thresholds, frame skip rates, and output options

## Pre-trained Models

This project uses two models from the [OpenVINO Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo):

- **`vehicle-detection-0200`**: Object detection model for identifying vehicles in images
  - Input: 256x256 RGB image
  - Output: Bounding boxes with confidence scores
  
- **`vehicle-attributes-recognition-barrier-0039`**: Classification model for vehicle attributes
  - Input: 72x72 RGB image (cropped vehicle)
  - Output: Color and type predictions

Models are automatically downloaded on first use.

## Installation

### Prerequisites

- Python 3.8+
- pip
- venv

### Create a Virtual Environment and Activate it

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate    # On Windows
```

### Install Dependencies

```bash
pip install openvino>=2023.1.0 opencv-python numpy tqdm requests
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## Project Structure

```
vehicle-detection-and-recognition/
├── vehicle_detection_recognition.py  # Core detection/recognition class
├── process_video.py                              # Video processing script
├── README.md                                     # This file
└── model/                                        # Auto-downloaded models (created on first run)
```

## Usage

### 1. Process Images

Use the core module to process individual images:

```python
from vehicle_detection_recognition import VehicleDetectionRecognition
import cv2

# Initialize pipeline (models auto-download if needed)
pipeline = VehicleDetectionRecognition(
    detection_model_path="model/vehicle-detection-0200.xml",
    recognition_model_path="model/vehicle-attributes-recognition-barrier-0039.xml",
    device="CPU"
)

# Load and process image
image = cv2.imread("car.jpg")
output_image, detections = pipeline.process_image(image, threshold=0.6)

# Print results
for det in detections:
    print(f"{det['color']} {det['type']} at {det['bbox']}")

# Save annotated image
cv2.imwrite("output.jpg", output_image)
```

Or run the example script:

```bash
python vehicle_detection_recognition.py
```

### 2. Process Videos

Process video files with automatic frame extraction:

#### Basic Usage

```bash
# Process all frames
python process_video.py video.mp4

# Process every 5th frame (5x faster)
python process_video.py video.mp4 --frame-skip 5
```

#### Advanced Options

```bash
# Custom output directory
python process_video.py video.mp4 -o my_detections --frame-skip 3

# Save output video + frames
python process_video.py video.mp4 --save-video --frame-skip 2

# Adjust detection threshold (0.0-1.0)
python process_video.py video.mp4 --threshold 0.7 --frame-skip 5

# Use GPU for inference
python process_video.py video.mp4 --device GPU

# Full example
python process_video.py traffic.mp4 \
    --output-dir traffic_detections \
    --frame-skip 10 \
    --threshold 0.65 \
    --save-video \
    --device CPU
```

#### Command-line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `video` | - | Required | Path to input video file (mp4, avi, etc.) |
| `--output-dir` | `-o` | `detected_frames` | Directory to save detected frames |
| `--frame-skip` | `-s` | `1` | Process every Nth frame (1=all frames) |
| `--threshold` | `-t` | `0.6` | Detection confidence threshold (0.0-1.0) |
| `--device` | `-d` | `CPU` | Device for inference (CPU, GPU, AUTO) |
| `--save-video` | - | `False` | Save output as video file |
| `--detection-model` | - | Auto-download | Path to custom detection model |
| `--recognition-model` | - | Auto-download | Path to custom recognition model |

### 3. Get Help

```bash
python process_video.py --help
```

## Output

### Saved Frames

Frames with detected vehicles are saved with descriptive filenames:

```
detected_frames/
├── frame_000000_vehicles_4.jpg  # Frame 0 with 4 vehicles
├── frame_000005_vehicles_3.jpg  # Frame 5 with 3 vehicles
├── frame_000010_vehicles_2.jpg  # Frame 10 with 2 vehicles
└── ...
```

### Console Output

```
Video: traffic.mp4
  Resolution: 1920x1080
  FPS: 30
  Total frames: 3333
  Frame skip: 10 (processing every 10 frame(s))
  Output directory: detected_frames

Processing video...
Frames: 100%|████████| 3333/3333 [00:11<00:00, 290.03frame/s]

✓ Processing complete!
  Total frames: 3333
  Processed frames: 334
  Frames with vehicles: 163
  Saved to: detected_frames
```

## Performance Tips

1. **Frame Skipping**: Use `--frame-skip` to process fewer frames
   - `--frame-skip 5`: Process every 5th frame (5x faster)
   - `--frame-skip 10`: Process every 10th frame (10x faster)

2. **Threshold Tuning**: Adjust `--threshold` to balance precision/recall
   - Higher (0.7-0.9): Fewer false positives, may miss some vehicles
   - Lower (0.4-0.6): More detections, may include false positives

3. **Device Selection**: Use GPU if available for faster inference
   ```bash
   python process_video.py video.mp4 --device GPU
   ```

## Examples

### Surveillance Footage

```bash
# Process parking lot surveillance (1 frame per second)
python process_video.py parking_lot.mp4 --frame-skip 30 --threshold 0.7
```

### Traffic Analysis

```bash
# Analyze highway traffic with high confidence
python process_video.py highway.mp4 --frame-skip 15 --threshold 0.75 --save-video
```

### Quick Preview

```bash
# Fast preview (every 20th frame)
python process_video.py test.mp4 --frame-skip 20 -o preview
```

## Troubleshooting

### Models Not Downloading

If automatic download fails, manually download models from:
- https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/

### Out of Memory

- Increase `--frame-skip` value
- Process shorter video segments
- Use lower resolution videos

### No Detections

- Lower `--threshold` value (try 0.4-0.5)
- Ensure video contains vehicles
- Check video quality and resolution

## License

This project uses OpenVINO™ and models from the Open Model Zoo. Please refer to their respective licenses.

## Acknowledgments

- [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino)
- [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
- Based on [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)