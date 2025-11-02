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

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()
client.simSetTraceLine([0.5, 0.0, 0.5, 1.0], 2.0) 

yaw_target = 0

cams = ["3"] #Front CamID is 0
cam_windows = ["Down"]

def camera_thread(client, cams, cam_windows):
    # Create windows
    
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
                
                # Convert RGB to BGR for OpenCV
                img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(img_hsv, (48,0,0), (180,255,203))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6,6), np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [c for c in contours if cv2.contourArea(c) > 5000]

                for c in contours:
                    if cv2.contourArea(c) < 5000:
                        continue

                    # Straight box (green)
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0,255,0), 1)

                    # Compute orientation using moments
                    M = cv2.moments(c)
                    if M["mu20"] + M["mu02"] == 0:
                        continue
                    angle = 0.5 * np.arctan2(2*M["mu11"], M["mu20"] - M["mu02"])
                    angle_deg = np.degrees(angle)

                    # Get centroid
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])

                    # Draw a rotated (blue) box using the angle
                    rect = cv2.minAreaRect(c)
                    center, (w, h), _ = rect
                    rot_rect = ((cx, cy), (w, h), angle_deg)
                    box = cv2.boxPoints(rot_rect)
                    box = np.int32(box)
                    cv2.drawContours(img_rgb, [box], 0, (255,0,0), 2)

                    cv2.putText(img_rgb, f"{angle_deg:.1f}°", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

                cv2.imshow("Annotated", img_rgb)
                cv2.imshow("Mask (Filtered Road)", mask)

            
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



# airsim.wait_key('Press any key to move vehicle to (-3, 3, -3) at 5 m/s')
# client.moveToPositionAsync(-3, 3, -3, 5).join()

# client.hoverAsync().join()

# state = client.getMultirotorState()
# print("state: %s" % pprint.pformat(state))

# airsim.wait_key('Press any key to take images')
# # get camera images from the car
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
#     airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
#     airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
#     airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
# print('Retrieved images: %d' % len(responses))

# tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
# print ("Saving images to %s" % tmp_dir)
# try:
#     os.makedirs(tmp_dir)
# except OSError:
#     if not os.path.isdir(tmp_dir):
#         raise

# for idx, response in enumerate(responses):

#     filename = os.path.join(tmp_dir, str(idx))

#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
#     elif response.compress: #png format
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
#     else: #uncompressed array
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
#         img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
#         cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

# airsim.wait_key('Press any key to reset to original state')

# client.reset()
# client.armDisarm(False)

# # that's enough fun for now. let's quit cleanly
# client.enableApiControl(False)
