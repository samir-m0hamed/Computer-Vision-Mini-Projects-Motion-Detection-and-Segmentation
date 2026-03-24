import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os

# COCO Class Names
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Setup output directory
output_dir = os.path.join(script_dir, "live_segmentation_results")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load Mask R-CNN model
print("Loading Mask R-CNN model...")
model = maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()
print("Model loaded successfully!")

# Open camera
print("\nOpening camera...")
cap = cv2.VideoCapture(0)

# Check if camera opened
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("Camera opened successfully!")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Camera Resolution: {frame_width}x{frame_height}")
print(f"Frames Per Second: {fps} FPS")

# Setup VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_dir, "segmented_video.mp4")
out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 30, (frame_width, frame_height))

print(f"\nStarting Live Stream... Press 'q' to exit")
print("=" * 60)

frame_count = 0
total_detections = 0
process_every_n_frames = 2  # Process every 2 frames for speed

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to read frame")
            break
        
        frame_count += 1
        
        # Skip frames for speed
        if frame_count % process_every_n_frames != 0:
            continue
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (640, 480))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = F.to_tensor(frame_rgb).to(device)
        
        # Perform prediction
        with torch.no_grad():
            predictions = model([img_tensor])
        
        # Extract results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        masks = predictions[0]['masks'].cpu().numpy()
        
        # Filter by confidence threshold
        confidence_threshold = 0.5
        valid_indices = scores >= confidence_threshold
        
        boxes = boxes[valid_indices]
        labels = labels[valid_indices]
        scores = scores[valid_indices]
        masks = masks[valid_indices]
        
        # Count detected objects
        num_objects = len(boxes)
        total_detections += num_objects
        
        # Calculate scale for original frame
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]
        
        display_frame = frame.copy()
        
        # Different colors for each object
        colors = [
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 0, 0),      # Blue
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Purple
        ]
        
        for idx in range(num_objects):
            color = colors[idx % len(colors)]
            
            # Draw bounding box on original frame
            box = boxes[idx].astype(int)
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Get class name
            class_id = labels[idx]
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
            
            # Write object name and confidence
            confidence = scores[idx]
            label_text = f"{class_name}: {confidence:.2f}"
            cv2.putText(display_frame, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Apply mask overlay on original frame
            mask = masks[idx, 0] > 0.5
            mask_resized = cv2.resize(mask.astype(np.uint8)*255, (display_frame.shape[1], display_frame.shape[0])) > 127
            display_frame[mask_resized] = display_frame[mask_resized] * 0.6 + np.array(color) * 0.4
        
        # Add info text
        info_text = f"Frame: {frame_count} | Objects: {num_objects} | Total: {total_detections}"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame to video
        out.write(display_frame)
        
        # Display frame
        cv2.imshow("Live Segmentation - Mask R-CNN (Press Q to Exit)", display_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopping live stream...")
            break
        
        # Print update every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames | Total Detections: {total_detections}")

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  - Total Frames: {frame_count}")
    print(f"  - Total Objects Detected: {total_detections}")
    print(f"  - Average Objects per Frame: {total_detections / max(frame_count, 1):.2f}")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo saved to: {output_video_path}")
    print("Program finished successfully!")
