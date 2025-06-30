# Step 1: Install dependencies (only needed once)
!pip install inference opencv-python-headless --quiet

# Step 2: Import required modules
from inference import InferencePipeline
import cv2
import os

# Step 3: Output video configuration
INPUT_VIDEO = "Clear f1.mp4"               # Input video file path or 0 for webcam
OUTPUT_VIDEO = "output_video.mp4"            # Output file path
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')     # Video codec

# Global variables for video writing
video_writer = None
frame_width, frame_height = None, None

# Step 4: Callback function to handle predictions and write frames to video
def my_sink(result, video_frame):
    global video_writer, frame_width, frame_height

    if result.get("output_image") is not None:
        frame = result["output_image"].numpy_image

        # Initialize video writer once with correct frame size
        if video_writer is None:
            frame_height, frame_width = frame.shape[:2]

            output_dir = os.path.dirname(OUTPUT_VIDEO)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            video_writer = cv2.VideoWriter(
                OUTPUT_VIDEO,
                FOURCC,
                5.0,  # FPS
                (frame_width, frame_height)
            )

            if not video_writer.isOpened():
                print("❌ Error: Could not open VideoWriter.")
                video_writer = None
                return

        # Write frame to video
        video_writer.write(frame)

# Step 5: Initialize the inference pipeline
pipeline = InferencePipeline.init_with_workflow(
    api_key="****************",                  
    workspace_name="detection-utzod",                
    workflow_id="detect-count-and-visualize-5",     
    video_reference=INPUT_VIDEO,                    
    max_fps=5,                                        
    on_prediction=my_sink                            
)

# Step 6: Start pipeline and wait for it to finish
print(f"🔄 Running inference on '{INPUT_VIDEO}'...")
pipeline.start()
pipeline.join()

# Step 7: Release resources
if video_writer is not None:
    video_writer.release()
    print(f"✅ Inference complete. Output saved to '{OUTPUT_VIDEO}'")
else:
    print("⚠️ No video frames processed or output writer not initialized.")
