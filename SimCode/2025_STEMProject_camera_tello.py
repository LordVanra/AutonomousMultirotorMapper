#!/usr/bin/env python3
"""
Tello Camera Stream Viewer (Functional Version)
Connects to Tello and displays the live camera feed
"""

import socket
import threading
import time
import cv2

# Global variables
command_socket = None
tello_address = ('192.168.10.1', 8889)
video_port = 11111
streaming = False
current_frame = None


def send_command(command):
    try:
        command_socket.sendto(command.encode('utf-8'), tello_address)
        response, _ = command_socket.recvfrom(1024)
        response_text = response.decode('utf-8')
        print(f"Response: {response_text}")
        return response_text
    except socket.timeout:
        print("Response: TIMEOUT")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def stream_video():
    global streaming, current_frame
    
    # Use OpenCV to capture UDP stream
    cap = cv2.VideoCapture(f'udp://0.0.0.0:{video_port}')
    
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    
    print("Video stream started")
    
    while streaming:
        ret, frame = cap.read()
        if ret:
            current_frame = frame
        else:
            time.sleep(0.01)
    
    cap.release()
    print("Video stream stopped")

def initialize_connection():
    global command_socket
    
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    command_socket.bind(('', 9000))
    command_socket.settimeout(5.0)

def start_stream():
    global streaming
    
    print("Starting video stream...")
    response = send_command("streamon")
    
    if response == "ok":
        streaming = True
        
        # Start video receiving thread
        video_thread = threading.Thread(target=stream_video)
        video_thread.daemon = True
        video_thread.start()
        
        # Give stream time to start
        print("Waiting for video stream to initialize...")
        time.sleep(3)
        
        # Display video
        display_video()
    else:
        print("Failed to enable stream")


def stop_stream():
    global streaming, command_socket
    
    print("Stopping stream...")
    streaming = False
    
    send_command("streamoff")
    cv2.destroyAllWindows()
    
    if command_socket:
        command_socket.close()


def main():
    try:
        initialize_connection()
        start_stream()
    except KeyboardInterrupt:
        print("\n\nEmergency stop!")
        streaming = False
        send_command("land")
        stop_stream()
    except Exception as e:
        print(f"\nError occurred: {e}")
        streaming = False
        stop_stream()

#Shi highkey broke        
def display_video():
    global streaming, current_frame

    cv2.namedWindow("Tello Camera", cv2.WINDOW_NORMAL)
    screenshot_count = 0
    
    try:
        while streaming:
            if current_frame is not None:
                # Add info overlay
                display_frame = current_frame.copy()
                height, width = display_frame.shape[:2]
                
                # Add text overlay
                cv2.putText(display_frame, "Tello Camera Feed", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press Q to quit", 
                          (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 1)
                
                cv2.imshow("Tello Camera", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                print("STATUS CHECKS")
                send_command("battery?")
                send_command("sdk?")
                send_command("wifi?")
                
                print("Drone will take off in 3 seconds!")
                time.sleep(3)
                send_command("takeoff")
                time.sleep(5)  # Wait for takeoff to complete
                
                send_command("up 50")
                time.sleep(3)
                
                send_command("flip f")  # Forward flip
                time.sleep(3)
                
                send_command("land")
                time.sleep(5)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        stop_stream()



if __name__ == "__main__":
    main()