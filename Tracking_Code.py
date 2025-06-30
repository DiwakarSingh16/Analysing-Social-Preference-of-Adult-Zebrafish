# Step 1: Install dependencies (only needed once)
!pip install inference opencv-python-headless --quiet

from inference import InferencePipeline
import cv2
import os
import numpy as np

# Input/output video files
INPUT_VIDEO = "Opaque f2.mp4"
OUTPUT_VIDEO = "output_video.mp4"

# Codec for output video
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# Regions to count fish entries (x1, y1, x2, y2)
REGION_C = (275, 50, 320, 400)
REGION_D = (330, 50, 370, 400)

# Fish entry counters
count_C = 0
count_D = 0
counted_ids_C = set()
counted_ids_D = set()

# Global video writer and frame dims
video_writer = None
frame_width = None
frame_height = None

def to_float(val):
    # Convert numpy array scalar or float to float
    if isinstance(val, np.ndarray):
        return float(val.item())
    return float(val)

def my_sink(result, video_frame):
    global video_writer, frame_width, frame_height, count_C, count_D, counted_ids_C, counted_ids_D

    # Extract output image frame from pipeline
    if result.get("output_image") is not None:
        frame = result["output_image"].numpy_image

        # Initialize writer if needed
        if video_writer is None:
            frame_height, frame_width = frame.shape[:2]
            output_dir = os.path.dirname(OUTPUT_VIDEO)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO, FOURCC, 5.0, (frame_width, frame_height))
            if not video_writer.isOpened():
                print("❌ Could not open VideoWriter")
                video_writer = None
                return

        # Process each detection: result["predictions"] is a list of tuples or similar
        predictions = result.get("predictions", [])

        for det in predictions:
            # Defensive parsing of detection tuple:
            # Usually: (x, y, width, height, confidence, class_id, class_name, track_id)
            # Adjust indexes if needed depending on your workflow output format

            # If det is tuple/list with at least 4 elements for bbox
            try:
                x = to_float(det[0])
                y = to_float(det[1])
                w = to_float(det[2])
                h = to_float(det[3])
            except Exception as e:
                # If unexpected format, skip this detection
                continue

            # Calculate bbox coords
            left = int(x - w / 2)
            top = int(y - h / 2)
            right = int(x + w / 2)
            bottom = int(y + h / 2)

            # Get fish ID if tracking info exists (commonly 7th index)
            fish_id = None
            if len(det) > 7:
                try:
                    fish_id = int(det[7])
                except:
                    fish_id = None

            # Center point of bbox
            cx, cy = x, y ,ε=350,d=40  #cy-y 

            # Check if fish entered Region C
            if REGION_C[0] < cx < REGION_C[2] and REGION_C[1] < cy < REGION_C[3] ==ε:
                if fish_id is not None and fish_id not in counted_ids_C:
                    count_C += 1
                    counted_ids_C.add(fish_id)

            # Check if fish entered Region D
            if REGION_D[0] < cx < REGION_D[2] and REGION_D[1] < cy < REGION_D[3] ==ε:
                if fish_id is not None and fish_id not in counted_ids_D:
                    count_D += 1
                    counted_ids_D.add(fish_id)

            # Draw bbox and label on frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Use class name if available at index 6, else generic label
            label = str(det[6]) if len(det) > 6 else "Fish"
            cv2.putText(frame, label, (left, max(top - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw regions and counts
        cv2.rectangle(frame, REGION_C[:2], REGION_C[2:], (0, 165, 255), 2)
        cv2.putText(frame, f"Region C: {count_C}", (REGION_C[0], max(REGION_C[1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.rectangle(frame, REGION_D[:2], REGION_D[2:], (0, 0, 255), 2)
        cv2.putText(frame, f"Region D: {count_D}", (REGION_D[0], max(REGION_D[1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Write frame to video
        if video_writer is not None:
            video_writer.write(frame)

# Initialize pipeline
pipeline = InferencePipeline.init_with_workflow(
    api_key="*****************",
    workspace_name="detection-utzod",
    workflow_id="detect-count-and-visualize-2",
    video_reference=INPUT_VIDEO,
    max_fps=5,
    on_prediction=my_sink
)

print(f"🔄 Running inference on '{INPUT_VIDEO}'...")
pipeline.start()
pipeline.join()

if video_writer is not None:
    video_writer.release()
    print(f"✅ Inference complete. Output saved to '{OUTPUT_VIDEO}'")
else:
    print("⚠️ No frames processed or video writer not initialized.")
