import cv2
import numpy as np
import os
import sys
import importlib.util

# Ensure we are in the script's directory so that YOLO can find 'model_weights.pt' unless it's an absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory set to: {script_dir}")

# Add YOLO directory to path
yolo_dir = os.path.join(script_dir, 'YOLO')
sys.path.append(yolo_dir)

# Import YOLO functions (module name starts with a number, so use importlib)
yolo_model_path = os.path.join(yolo_dir, '2025_STEMProject_YOLO_Model.py')
spec = importlib.util.spec_from_file_location("yolo_model", yolo_model_path)
yolo_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_model)



def main():
    VIDEO_PATH = 'output.avi' 
    OUTPUT_PATH = 'output_video.mp4' 
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}/{total_frames}...")

            # Run YOLO prediction
            result = yolo_model.predict(frame)
            
            frame_to_write = frame
            
            if result is not None:
                annotated_frame, _ = yolo_model.visualize_segmentation(result, output_path=None)
                frame_to_write = annotated_frame
                
                cv2.imshow('YOLO Video Analysis', annotated_frame)
            else:
                cv2.imshow('YOLO Video Analysis', frame)

            # Ensure frame size matches video writer expectations
            if frame_to_write.shape[1] != width or frame_to_write.shape[0] != height:
                frame_to_write = cv2.resize(frame_to_write, (width, height))

            out.write(frame_to_write)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video processing complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
