import setup_path
import airsim

import numpy as np
import pprint
import threading
import cv2
import keyboard 
import time
import math
import sys
import os
import importlib.util

# Add YOLO directory to path
yolo_dir = os.path.join(os.path.dirname(__file__), 'YOLO')
sys.path.append(yolo_dir)

# Import YOLO functions (module name starts with number, so use importlib)
yolo_model_path = os.path.join(yolo_dir, '2025_STEMProject_YOLO_Model.py')
spec = importlib.util.spec_from_file_location("yolo_model", yolo_model_path)
yolo_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_model)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

camera_client = airsim.MultirotorClient()
camera_client.confirmConnection()

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()
client.simSetTraceLine([0.5, 0.0, 0.5, 1.0], 2.0) 

yaw_target = 0

cams = ["0"] #Front CamID is 0
cam_windows = ["Front"]

def camera_thread(client, cams, cam_windows):
    while True:
        try:
            # Request camera images
            responses = camera_client.simGetImages([
                airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False) 
                for cam in cams
            ])
            
            # Process each response
            for i, response in enumerate(responses):
                # Skip invalid responses
                if response.width == 0 or response.height == 0:
                    print(f"Skipping invalid response: width={response.width}, height={response.height}")
                    continue
                
                img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img = img_data.reshape(response.height, response.width, 3)

                result = yolo_model.predict(img)
    
                if result is not None:
                    yolo_model.visualize_segmentation(result, "output.jpg")
                
                cv2.imshow("Annotated", img)
                cv2.waitKey(1) 
            
            if keyboard.is_pressed('z') and keyboard.is_pressed('x'):
                break
                
        except Exception as e:
            print(f"Error in camera thread: {e}")
            break

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