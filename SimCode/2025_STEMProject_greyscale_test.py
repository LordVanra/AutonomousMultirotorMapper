from __future__ import print_function
import setup_path
import airsim

import numpy as np
import pprint
import threading
import cv2 as cv
import keyboard 
import argparse
import time
import math

import random as rng
rng.seed(12345)

def thresh_callback(val):
    threshold = val
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    
    # Fix for OpenCV 4.x - findContours returns only 2 values
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    
    cv.imshow('Contours', drawing)

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

cams = ["3"] #Front CamID is 0
cam_windows = ["Down"]

def camera_thread(client, cams, cam_windows):
    global src_gray
    
    # Create window and trackbar once before the loop
    source_window = 'Source'
    cv.namedWindow(source_window)
    max_thresh = 255
    thresh = 100
    cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
    
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
                
                # Reshape the image properly
                img1d = img1d.reshape(response.height, response.width, -1)
                
                # Convert image to gray if it's not already
                if len(img1d.shape) == 3 and img1d.shape[2] == 3:
                    # Image is BGR, convert to grayscale
                    src_gray = cv.cvtColor(img1d, cv.COLOR_BGR2GRAY)
                elif len(img1d.shape) == 3 and img1d.shape[2] == 4:
                    # Image is BGRA, convert to grayscale
                    src_gray = cv.cvtColor(img1d, cv.COLOR_BGRA2GRAY)
                else:
                    # Image is already grayscale
                    src_gray = img1d if len(img1d.shape) == 2 else img1d[:,:,0]
                
                src_gray = cv.blur(src_gray, (3,3))
                
                # Just display the image - trackbar already exists
                cv.imshow(source_window, src_gray)
                
                # Get current trackbar position and call callback
                current_thresh = cv.getTrackbarPos('Canny thresh:', source_window)
                thresh_callback(current_thresh)

            # Check for quit key
            if keyboard.is_pressed('z') and keyboard.is_pressed('x'):
                break
            if cv.waitKey(1) & 0xFF == 27:  # ESC to exit normally
                break
                
        except Exception as e:
            print(f"Error in camera thread: {e}")
            break
    
    cv.destroyAllWindows()

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

cv.destroyAllWindows()
cam_thread.join()
client.armDisarm(False)
client.reset()
client.enableApiControl(False)

