import setup_path
import airsim

import numpy as np
import pprint
import threading
import cv2
import keyboard 
import time
import math

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

camera_client = airsim.MultirotorClient()
camera_client.confirmConnection()

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

barometer_data = client.getBarometerData()
s = pprint.pformat(barometer_data)
print("barometer_data: %s" % s)

magnetometer_data = client.getMagnetometerData()
s = pprint.pformat(magnetometer_data)
print("magnetometer_data: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

# Load YOLO model for object detection
print("Loading YOLO model...")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("YOLO model loaded successfully!")

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()
client.simSetTraceLine([0.5, 0.0, 0.5, 1.0], 2.0) 

yaw_target = 0

cams = ["3"] #Down camera
cam_windows = ["Down"]

def camera_thread(client, cams, cam_windows):
    
    while True:
        try:
            # Request all camera images at once
            responses = camera_client.simGetImages([
                airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False) 
                for cam in cams
            ])
            
            # Process and display each response
            for i, response in enumerate(responses):
                # Skip invalid responses
                if response.width == 0 or response.height == 0:
                    continue
                
                # Convert to numpy array
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                
                # Reshape to 2D image (height x width x channels)
                img_rgb = img1d.reshape(response.height, response.width, 3).copy()
                
                height, width, channels = img_rgb.shape
                
                # Prepare image for YOLO
                blob = cv2.dnn.blobFromImage(img_rgb, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)
                
                # Process detections
                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        # Filter for road-related classes with confidence > 0.5
                        # COCO classes: road is not directly labeled, but we can use street, traffic light, etc.
                        # You may need to adjust this based on what YOLO detects
                        if confidence > 0.3:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                # Apply non-max suppression
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
                
                # Draw bounding boxes
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = confidences[i]
                        
                        # Draw rectangle
                        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img_rgb, f"{label}: {confidence:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imshow("Annotated", img_rgb)
            
            # Check for quit key
            if keyboard.is_pressed('z') and keyboard.is_pressed('x'):
                camera_running = False
                break
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit normally
                break
                
        except Exception as e:
            print(f"Error in camera thread: {e}")
            break
    
    cv2.destroyAllWindows()

# # Start camera thread
cam_thread = threading.Thread(target=camera_thread, args=(camera_client, cams, cam_windows))
cam_thread.daemon = True
cam_thread.start()

while True:
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    x, y, z = pos.x_val, pos.y_val, pos.z_val
    roll, pitch, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
    vx, vy, vz = 0, 0, 0

    # Forward/backward
    if keyboard.is_pressed('p'):
        keyboard.wait('p')
        client.moveToPositionAsync(float(input("Enter X: ")), float(input("Enter Y: ")), z, 2).join()

    # WASD movement relative to yaw
    if keyboard.is_pressed('w'):
        vx += math.cos(yaw)
        vy += math.sin(yaw)
    if keyboard.is_pressed('s'):
        vx -= math.cos(yaw)
        vy -= math.sin(yaw)
    if keyboard.is_pressed('a'):
        vx += math.sin(yaw)
        vy -= math.cos(yaw)
    if keyboard.is_pressed('d'):
        vx -= math.sin(yaw)
        vy += math.cos(yaw)

    # Rotate yaw
    if keyboard.is_pressed('q'):
        yaw_target -= 2  
    elif keyboard.is_pressed('e'):
        yaw_target += 2

    # Altitude control
    if keyboard.is_pressed('x') and z > -50:
        z -= 0.5
    elif keyboard.is_pressed('c') and z < 0:
        z += 0.5

    # Send velocity command
    client.moveByVelocityZAsync(
        3*vx, 3*vy, z, 0.1,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, yaw_target)
    )

    # Quit
    if keyboard.is_pressed('z') and keyboard.is_pressed('x'):
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        break

    time.sleep(0.05)

cv2.destroyAllWindows()
cam_thread.join()
client.armDisarm(False)
client.reset()
client.enableApiControl(False)