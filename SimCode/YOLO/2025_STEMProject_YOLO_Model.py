from ultralytics import YOLO
import cv2
import numpy as np

def predict(model, image_path):
    configs = [
        {'conf': 0.3, 'imgsz': 640},
        {'conf': 0.2, 'imgsz': 640},
        {'conf': 0.1, 'imgsz': 640}
    ]
    
    best_result = None
    best_config = None
    
    for i, config in enumerate(configs):
        print(f"\nTrying config {i+1}: {config}")
        results = model.predict(source=image_path, **config)
        result = results[0]
        
        has_masks = result.masks is not None and len(result.masks) > 0
        has_boxes = result.boxes is not None and len(result.boxes) > 0
        
        print(f"  Masks: {len(result.masks) if has_masks else 0}")
        print(f"  Boxes: {len(result.boxes) if has_boxes else 0}")
        
        if has_masks or has_boxes:
            if best_result is None:
                best_result = result
                best_config = config
            print(f"  ✓ Detection successful")
    
    if best_result is None:
        print("\n✗ No detections with any configuration")
    else:
        print(f"\nBest result with config: {best_config}")
    
    return best_result

def visualize_segmentation(result, output_path='result.jpg'):
    """Visualize segmentation results with overlay and also output BW mask"""
    if result.masks is not None:
        original_img = result.orig_img.copy()
        masks = result.masks.data.cpu().numpy()

        # combined binary mask 
        combined_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)

        overlay = np.zeros_like(original_img)
        for mask in masks:
            mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            combined_mask[binary_mask == 1] = 255
            overlay[binary_mask == 1] = [0, 255, 0]

        alpha = 0.5
        result_image = cv2.addWeighted(original_img, 1, overlay, alpha, 0)
        
        # Calculate center of each white line per row and draw red centerline
        centerline_points = []
        for row_idx in range(combined_mask.shape[0]):
            row = combined_mask[row_idx]
            white_pixels = np.where(row == 255)[0]
            
            if len(white_pixels) > 0:
                # Find the center of the white line in this row
                center_x = int(np.mean(white_pixels))
                centerline_points.append((center_x, row_idx))
        
        cv2.imwrite(output_path, result_image)
        print(f"Segmentation result saved as '{output_path}'")

        bw_path = "output_mask.jpg"
        cv2.imwrite(bw_path, combined_mask)
        print(f"Black/white road mask saved as '{bw_path}'")

    else:
        # If no masks, fall back to default plot
        result_image = result.plot()
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image_bgr)
        print(f"No segmentation masks found. Saved default plot as '{output_path}'")

def main():
    # Configuration
    MODEL_PATH = 'model_weights.pt'
    IMAGE_PATH = 'input_frame.png' 
    OUTPUT_PATH = 'output_colored.jpg'
    
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    print(f"\nProcessing image: {IMAGE_PATH}")
    
    result = predict(model, IMAGE_PATH)
    
    if result is not None:
        # Visualize and save results
        visualize_segmentation(result, OUTPUT_PATH)
        
        # Print additional information
        if result.masks is not None:
            print(f"\nDetected {len(result.masks)} road segment(s)")
            
            # Get confidence scores if available
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                print(f"Confidence scores: {confidences}")
    else:
        print("Failed to detect road in the image.")

if __name__ == "__main__":
    main()