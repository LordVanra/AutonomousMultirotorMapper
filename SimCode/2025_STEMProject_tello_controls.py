from djitellopy import tello

def emergency_land():
    try:
        drone.land()
    except Exception as E:
        emergency_land()

try:

    drone = tello.Tello()
    drone.connect()

    drone.takeoff()

    drone.rotate_clockwise(260)

    drone.flip("f")

    drone.land()

    drone.end()

except Exception as E:
    emergency_land()