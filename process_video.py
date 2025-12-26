"""
Process video files for vehicle detection and recognition
Saves frames with detected vehicles to an output directory
"""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from vehicle_detection_recognition import VehicleDetectionRecognition, download_models


def process_video(
    video_path,
    output_dir="images",
    frame_skip=1,
    threshold=0.6,
    detection_model_path=None,
    recognition_model_path=None,
    device="CPU",
    save_video=False
):
    """
    Process video file for vehicle detection and recognition
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save detected frames
        frame_skip: Process every Nth frame (1 = process all frames)
        threshold: Detection confidence threshold
        detection_model_path: Path to detection model (auto-downloads if None)
        recognition_model_path: Path to recognition model (auto-downloads if None)
        device: Device to run inference on (CPU, GPU, etc.)
        save_video: Whether to save output as video file
    
    Returns:
        Number of frames processed
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download models if not provided
    if detection_model_path is None or recognition_model_path is None:
        print("Downloading models...")
        detection_model_path, recognition_model_path = download_models()
    
    # Initialize pipeline
    print("Initializing vehicle detection pipeline...")
    pipeline = VehicleDetectionRecognition(
        detection_model_path,
        recognition_model_path,
        device=device
    )
    
    # Open video
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frame skip: {frame_skip} (processing every {frame_skip} frame(s))")
    print(f"  Output directory: {output_dir}")
    
    # Setup video writer if saving output video
    video_writer = None
    if save_video:
        output_video_path = output_dir / f"{video_path.stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps / frame_skip,  # Adjust FPS based on frame skip
            (width, height)
        )
        print(f"  Output video: {output_video_path}")
    
    # Process video
    frame_count = 0
    processed_count = 0
    saved_count = 0
    
    print("\nProcessing video...")
    with tqdm(total=total_frames, desc="Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on frame_skip parameter
            if frame_count % frame_skip != 0:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Process frame
            output_frame, detections = pipeline.process_image(frame, threshold=threshold)
            processed_count += 1
            
            # Save frame if vehicles detected
            if len(detections) > 0:
                # Save annotated frame
                frame_filename = f"frame_{frame_count:06d}_vehicles_{len(detections)}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), output_frame)
                saved_count += 1
                
                # Update progress bar with detection info
                pbar.set_postfix({
                    'detected': len(detections),
                    'saved': saved_count
                })
            
            # Write to output video if enabled
            if video_writer is not None:
                video_writer.write(output_frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    
    print(f"\n✓ Processing complete!")
    print(f"  Total frames: {frame_count}")
    print(f"  Processed frames: {processed_count}")
    print(f"  Frames with vehicles: {saved_count}")
    print(f"  Saved to: {output_dir}")
    
    return processed_count


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Process video files for vehicle detection and recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (mp4, avi, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="images",
        help="Directory to save detected frames"
    )
    
    parser.add_argument(
        "-s", "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames, 5 = every 5th frame)"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.6,
        help="Detection confidence threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output as video file (in addition to frames)"
    )
    
    parser.add_argument(
        "--detection-model",
        type=str,
        default=None,
        help="Path to detection model (.xml). Auto-downloads if not specified"
    )
    
    parser.add_argument(
        "--recognition-model",
        type=str,
        default=None,
        help="Path to recognition model (.xml). Auto-downloads if not specified"
    )
    
    args = parser.parse_args()
    
    # Process video
    try:
        process_video(
            video_path=args.video,
            output_dir=args.output_dir,
            frame_skip=args.frame_skip,
            threshold=args.threshold,
            detection_model_path=args.detection_model,
            recognition_model_path=args.recognition_model,
            device=args.device,
            save_video=args.save_video
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
