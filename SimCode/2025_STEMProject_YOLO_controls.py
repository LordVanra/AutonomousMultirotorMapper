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
import matplotlib.pyplot as plt

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

auto_vx = 0.0
auto_yaw_rate = 0.0
line_detected = False
lock = threading.Lock()

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
                    annotated_img, path_points = yolo_model.visualize_segmentation(result, "output.jpg")
                    
                    with lock:
                        if len(path_points) > 0:
                            lower_points = [p for p in path_points if p[1] > response.height // 2]
                            if not lower_points:
                                lower_points = path_points
                            
                            avg_x = sum(p[0] for p in lower_points) / len(lower_points)
                            center_x = response.width / 2
                            error_x = (avg_x - center_x) / (response.width / 2) # Normalized -1 to 1
                            
                            global auto_vx, auto_yaw_rate, line_detected
                            line_detected = True
                            auto_vx = 1.0 
                            auto_yaw_rate = error_x * 30 
                        else:
                            line_detected = False
                            auto_vx = 0.0
                            auto_yaw_rate = 0.0
                else:
                    with lock:
                        line_detected = False
                
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

path_x = []
path_y = []

while True:
    try:
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

        v_x_final, v_y_final = 3*vx, 3*vy
        
        with lock:
            if line_detected:
                v_x_final = auto_vx * math.cos(yaw)
                v_y_final = auto_vx * math.sin(yaw)
                yaw_target += auto_yaw_rate * 0.05 

        # Send velocity command
        client.moveByVelocityZAsync(
            v_x_final, v_y_final, z, 0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, yaw_target)
        )

        path_x.append(x)
        path_y.append(y)

        # Quit
        if keyboard.is_pressed('z') and keyboard.is_pressed('x'):
            print("Landing...")
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            break

        time.sleep(0.05)
    except KeyboardInterrupt:
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        break

plt.figure(figsize=(10, 10))
plt.plot(path_x, path_y, label='Drone Path', color='blue', linewidth=2)
plt.scatter(path_x[0], path_y[0], color='green', label='Start', zorder=5)
plt.scatter(path_x[-1], path_y[-1], color='red', label='End', zorder=5)
plt.title('Drone Movement Path')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('movement_path_yolo.png')
print("Movement path saved to movement_path_yolo.png")

cv2.destroyAllWindows()
cam_thread.join()
client.armDisarm(False)
client.reset()
client.enableApiControl(False)