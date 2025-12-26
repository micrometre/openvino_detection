"""
Simplified Vehicle Detection and Recognition using OpenVINO
Detects vehicles in images and recognizes their attributes (color and type)
"""

import os
from pathlib import Path
import cv2
import numpy as np
import openvino as ov


class VehicleDetectionRecognition:
    """Vehicle detection and recognition pipeline using OpenVINO models"""
    
    def __init__(self, detection_model_path, recognition_model_path, device="CPU"):
        """
        Initialize the vehicle detection and recognition models
        
        Args:
            detection_model_path: Path to vehicle detection model (.xml)
            recognition_model_path: Path to vehicle attributes recognition model (.xml)
            device: Device to run inference on (CPU, GPU, etc.)
        """
        self.core = ov.Core()
        self.device = device
        
        # Load detection model
        self.detection_model = self._load_model(detection_model_path)
        self.detection_input = self.detection_model.input(0)
        self.detection_output = self.detection_model.output(0)
        self.detection_shape = list(self.detection_input.shape)[2:]  # [height, width]
        
        # Load recognition model
        self.recognition_model = self._load_model(recognition_model_path)
        self.recognition_input = self.recognition_model.input(0)
        self.recognition_shape = list(self.recognition_input.shape)[2:]  # [height, width]
        
        # Vehicle attributes
        self.colors = ["White", "Gray", "Yellow", "Red", "Green", "Blue", "Black"]
        self.types = ["Car", "Bus", "Truck", "Van"]
    
    def _load_model(self, model_path):
        """Load and compile an OpenVINO model"""
        model = self.core.read_model(model=model_path)
        compiled_model = self.core.compile_model(model=model, device_name=self.device)
        return compiled_model
    
    def preprocess_image(self, image, target_shape):
        """
        Preprocess image for model inference
        
        Args:
            image: Input image (BGR format)
            target_shape: Target shape [height, width]
        
        Returns:
            Preprocessed image in shape [1, 3, H, W]
        """
        resized = cv2.resize(image, (target_shape[1], target_shape[0]))
        transposed = resized.transpose(2, 0, 1)  # HWC to CHW
        batched = np.expand_dims(transposed, 0)  # Add batch dimension
        return batched
    
    def detect_vehicles(self, image, threshold=0.6):
        """
        Detect vehicles in an image
        
        Args:
            image: Input image (BGR format)
            threshold: Confidence threshold for detections
        
        Returns:
            List of bounding boxes [x_min, y_min, x_max, y_max]
        """
        # Preprocess image
        input_image = self.preprocess_image(image, self.detection_shape)
        
        # Run inference
        boxes = self.detection_model([input_image])[self.detection_output]
        
        # Process detections
        boxes = np.squeeze(boxes, (0, 1))  # Remove batch and sequence dims
        boxes = boxes[~np.all(boxes == 0, axis=1)]  # Remove zero-only boxes
        
        # Convert normalized coordinates to absolute positions
        vehicle_boxes = self._boxes_to_positions(
            image, 
            cv2.resize(image, (self.detection_shape[1], self.detection_shape[0])),
            boxes, 
            threshold
        )
        
        return vehicle_boxes
    
    def _boxes_to_positions(self, original_image, resized_image, boxes, threshold):
        """
        Convert detection boxes to absolute positions in original image
        
        Args:
            original_image: Original image
            resized_image: Resized image used for detection
            boxes: Detection boxes from model
            threshold: Confidence threshold
        
        Returns:
            List of absolute bounding boxes
        """
        # Calculate scaling ratios
        orig_h, orig_w = original_image.shape[:2]
        resized_h, resized_w = resized_image.shape[:2]
        ratio_x = orig_w / resized_w
        ratio_y = orig_h / resized_h
        
        # Extract boxes (skip image_id and label columns)
        boxes = boxes[:, 2:]
        
        positions = []
        for box in boxes:
            conf = box[0]
            if conf > threshold:
                # Convert normalized coordinates to absolute positions
                x_min, y_min, x_max, y_max = box[1:]
                
                x_min = int(x_min * resized_w * ratio_x)
                y_min = int(max(y_min * resized_h * ratio_y, 10))  # Min 10 to keep visible
                x_max = int(x_max * resized_w * ratio_x)
                y_max = int(y_max * resized_h * ratio_y)
                
                # Validate bounding box
                # Clamp coordinates to image boundaries
                x_min = max(0, min(x_min, orig_w - 1))
                y_min = max(0, min(y_min, orig_h - 1))
                x_max = max(0, min(x_max, orig_w))
                y_max = max(0, min(y_max, orig_h))
                
                # Ensure valid box (min size of 10x10 pixels)
                if x_max > x_min + 10 and y_max > y_min + 10:
                    positions.append([x_min, y_min, x_max, y_max])
        
        return positions
    
    def recognize_vehicle(self, vehicle_image):
        """
        Recognize vehicle attributes (color and type)
        
        Args:
            vehicle_image: Cropped vehicle image (BGR format)
        
        Returns:
            Tuple of (color, type)
        """
        # Preprocess image
        input_image = self.preprocess_image(vehicle_image, self.recognition_shape)
        
        # Run inference
        color_output = self.recognition_model([input_image])[self.recognition_model.output(1)]
        type_output = self.recognition_model([input_image])[self.recognition_model.output(0)]
        
        # Process outputs
        color_probs = np.squeeze(color_output, (2, 3))
        type_probs = np.squeeze(type_output, (2, 3))
        
        # Get predictions
        color = self.colors[np.argmax(color_probs)]
        vehicle_type = self.types[np.argmax(type_probs)]
        
        return color, vehicle_type
    
    def process_image(self, image, threshold=0.6, draw_boxes=True):
        """
        Detect and recognize vehicles in an image
        
        Args:
            image: Input image (BGR format)
            threshold: Detection confidence threshold
            draw_boxes: Whether to draw bounding boxes and labels
        
        Returns:
            Tuple of (annotated_image, detections)
            detections: List of dicts with keys: bbox, color, type
        """
        # Detect vehicles
        vehicle_boxes = self.detect_vehicles(image, threshold)
        
        # Recognize each vehicle
        detections = []
        output_image = image.copy()
        
        for bbox in vehicle_boxes:
            x_min, y_min, x_max, y_max = bbox
            
            # Crop vehicle region
            vehicle_crop = image[y_min:y_max, x_min:x_max]
            
            # Validate crop is not empty
            if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 10 or vehicle_crop.shape[1] < 10:
                continue  # Skip invalid crops
            
            try:
                # Recognize attributes
                color, vehicle_type = self.recognize_vehicle(vehicle_crop)
                
                detections.append({
                    'bbox': bbox,
                    'color': color,
                    'type': vehicle_type
                })
                
                # Draw annotations if requested
                if draw_boxes:
                    # Draw bounding box (red)
                    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    
                    # Draw label (green text)
                    label = f"{color} {vehicle_type}"
                    cv2.putText(
                        output_image,
                        label,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
            except Exception as e:
                # Skip this detection if recognition fails
                continue
        
        return output_image, detections


def download_models(base_dir="model", precision="FP32"):
    """
    Download vehicle detection and recognition models from OpenVINO Model Zoo
    
    Args:
        base_dir: Directory to save models
        precision: Model precision (FP32, FP16, FP16-INT8)
    
    Returns:
        Tuple of (detection_model_path, recognition_model_path)
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)
    
    detection_model = "vehicle-detection-0200"
    recognition_model = "vehicle-attributes-recognition-barrier-0039"
    base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
    
    detection_path = base_dir / f"{detection_model}.xml"
    recognition_path = base_dir / f"{recognition_model}.xml"
    
    # Download detection model
    if not detection_path.exists():
        print(f"Downloading {detection_model}...")
        for ext in [".xml", ".bin"]:
            url = f"{base_url}/{detection_model}/{precision}/{detection_model}{ext}"
            _download_file(url, base_dir / f"{detection_model}{ext}")
    
    # Download recognition model
    if not recognition_path.exists():
        print(f"Downloading {recognition_model}...")
        for ext in [".xml", ".bin"]:
            url = f"{base_url}/{recognition_model}/{precision}/{recognition_model}{ext}"
            _download_file(url, base_dir / f"{recognition_model}{ext}")
    
    return str(detection_path), str(recognition_path)


def _download_file(url, output_path):
    """Download a file from URL"""
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"  Downloaded: {output_path.name}")


def main():
    """Example usage"""
    # Download models if needed
    detection_model_path, recognition_model_path = download_models()
    
    # Initialize pipeline
    print("Initializing vehicle detection and recognition pipeline...")
    pipeline = VehicleDetectionRecognition(
        detection_model_path,
        recognition_model_path,
        device="CPU"
    )
    
    # Download test image
    test_image_path = Path("data/cars.jpg")
    if not test_image_path.exists():
        print("Downloading test image...")
        test_image_path.parent.mkdir(exist_ok=True)
        url = "https://storage.openvinotoolkit.org/data/test_data/images/person-bicycle-car-detection.bmp"
        _download_file(url, test_image_path)
    
    # Load and process image
    print(f"Processing image: {test_image_path}")
    image = cv2.imread(str(test_image_path))
    
    # Run detection and recognition
    output_image, detections = pipeline.process_image(image, threshold=0.6)
    
    # Print results
    print(f"\nDetected {len(detections)} vehicle(s):")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['color']} {det['type']} at {det['bbox']}")
    
    # Save output
    output_path = "output_vehicles.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
