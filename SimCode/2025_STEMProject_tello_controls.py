from djitellopy import Tello
import cv2
import time

def emergency_land():
    try:
        drone.land()
    except Exception as E:
        emergency_land()

try:
    drone = Tello()
    drone.connect()

    drone.takeoff()

    # Start video stream
    drone.streamon()
    frame_reader = drone.get_frame_read()

    # Get frame size for video writer
    frame = frame_reader.frame
    height, width, _ = frame.shape
    out = cv2.VideoWriter('drone_output.avi',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          20,  # fps
                          (width, height))

    start_time = time.time()
    duration = 10  

    drone.rotate_clockwise(260)
    drone.flip("f")

    while time.time() - start_time < duration:
        frame = frame_reader.frame
        out.write(frame)

    # Cleanup
    out.release()
    drone.land()
    drone.end()

except Exception as E:
    emergency_land()
