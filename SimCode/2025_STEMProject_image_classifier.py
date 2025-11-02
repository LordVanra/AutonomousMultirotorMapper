import setup_path
import airsim

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pprint
import threading
import cv2
import keyboard 
import time
import math

# load YOLOP from torch.hub
model = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def detect_road_area(img_bgr):
    # 1. Resize to YOLOP expected size (keep aspect ratio simple)
    img_resized = cv2.resize(img_bgr, (640, 640))
    img = img_resized / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = torch.from_numpy(img).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

    # Handle tuple or dict outputs
    if isinstance(outputs, (list, tuple)):
        seg = outputs[1]
    elif isinstance(outputs, dict) and 'drivable_area_segmentation' in outputs:
        seg = outputs['drivable_area_segmentation']
    else:
        raise RuntimeError(f"Unexpected YOLOP output type: {type(outputs)}")

    if seg.ndim == 4:
        seg = seg[0]
    if seg.shape[0] == 2:
        seg = seg[1]  # drivable area

    mask = torch.sigmoid(seg).cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))

    # --- Clean and smooth mask before overlay ---
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Overlay in red (only road)
    overlay = img_bgr.copy()
    overlay[mask == 255] = [0, 0, 255]
    combined = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)

    return combined, mask


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

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()
client.simSetTraceLine([0.5, 0.0, 0.5, 1.0], 2.0) 

yaw_target = 0

cams = ["3"] #Down camera
cam_windows = ["Down"]

def camera_thread(client, cams):
    while True:
        try:
            responses = camera_client.simGetImages([
                airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
                for cam in cams
            ])

            for i, response in enumerate(responses):
                if response.width == 0 or response.height == 0:
                    continue

                # Convert to RGB (AirSim images come as RGB bytes)
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)

                # Detect road areas
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                combined, road_mask = detect_road_area(img_bgr)

                # Show image in OpenCV window
                cv2.imshow("Down Camera (Road Detection)", combined)
                cv2.imshow("Down Camera Raw", img_rgb)

            # Quit keys
            if keyboard.is_pressed('z') and keyboard.is_pressed('x'):
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break

        except Exception as e:
            print(f"Error in camera thread: {e}")
            break

    cv2.destroyAllWindows()

# # Start camera thread
cam_thread = threading.Thread(target=camera_thread, args=(camera_client, cams))

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