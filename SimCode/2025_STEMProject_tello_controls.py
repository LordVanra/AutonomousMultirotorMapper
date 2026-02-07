from djitellopy import Tello
import cv2
import time

def sendCommand(command, maxTries=10):
    try:
        command()
    except Exception as E:
        if maxTries > 0:
            sendCommand(command, maxTries - 1)
        else:
            print(f"Error: {E}")
            raise

drone = None
out = None
frame_reader = None

try:
    drone = Tello()
    drone.connect()

    drone.streamon()
    time.sleep(2)

    frame_reader = drone.get_frame_read()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (960, 720))

    drone.takeoff()
    time.sleep(2)

    # Check battery
    battery = drone.get_battery()
    print(f"Battery level: {battery}%")

    start_time = time.time()
    rotate_done = False
    flip_done = False

    while time.time() - start_time < 15: 
        frame = frame_reader.frame
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        current_time = time.time() - start_time
        
        if current_time > 2 and not rotate_done:
            sendCommand(lambda: drone.rotate_clockwise(260))
            rotate_done = True
        
        if current_time > 5 and not flip_done:
            sendCommand(lambda: drone.flip("f"))
            flip_done = True
        
        time.sleep(0.033)

    drone.land()
    out.release()
    drone.streamoff()
    drone.end()

except Exception as E:
    print(f"Error: {E}")
    if drone:
        try:
            drone.land()
        except:
            pass
    if out:
        out.release()
    if drone:
        try:
            drone.streamoff()
            drone.end()
        except:
            pass