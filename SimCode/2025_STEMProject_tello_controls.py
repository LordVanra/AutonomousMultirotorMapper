import threading
from djitellopy import Tello
import cv2
import time

def sendCommand(command, maxTries=2):
    try:
        command()
    except Exception as E:
        if maxTries > 0:
            sendCommand(command, maxTries - 1)
        else:
            print(f"Error: {E}")

drone = None
out = None
frame_reader = None
stop_recording = False

def video_recording_thread():
    global stop_recording, out, frame_reader

    while not stop_recording:
        try:
            frame = frame_reader.frame
            if frame is not None:
                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            time.sleep(1/30) # Record at approximately 30 FPS
        except Exception as e:
            print(f"Error in video recording thread: {e}")
            break

    print("Video recording thread stopped")

try:
    drone = Tello()
    drone.connect()

    drone.streamon()
    time.sleep(2)

    frame_reader = drone.get_frame_read()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (960, 720))

    # Start video recording in a separate thread
    recording_thread = threading.Thread(target=video_recording_thread)
    recording_thread.start()

    # Movement commands
    drone.takeoff()
    time.sleep(2)

    # Check battery
    battery = drone.get_battery()
    print(f"Battery level: {battery}%")

    sendCommand(lambda: drone.move_up(50))
        
    time.sleep(5)

    print("Landing...")
    drone.land()

    # Stop recording and wait for thread to finish
    stop_recording = True
    recording_thread.join()

    out.release()
    drone.streamoff()
    drone.end()

except Exception as E:
    print(f"Error: {E}")
    stop_recording = True
    if 'recording_thread' in locals() and recording_thread.is_alive():
        recording_thread.join()
        
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