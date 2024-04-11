import random

from matplotlib import pyplot as plt

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging, Level
from beamngpy.sensors import Camera, Damage, Electrics, GForces, Timer, Ultrasonic
import time
import shutil
import os
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Camera, Lidar
from desktopmagic.screengrab_win32 import getDisplayRects, getRectAsImage

BEAMNG_GAME_PATH_DIR = "E:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"
BEAMNG_TECH_GAME_PATH_DIR = "E:/Program Files (x86)/Games/BeamNG.tech.v0.31.3.0"
BEAMNG_TECH_GAME_PATH_DIR = "C:/Users/laptop/Desktop/BeamNG.tech.v0.31.3.0"
SCREENSHOT_FOLDER_PATH = '../screenshots/'
CAMERA_FOLDER_PATH = '../camera/'
LIDAR_FOLDER_PATH = '../lidar/'
SCREENS = (getDisplayRects())


def data_folder_check(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Removed directory {folder_path}")
            os.makedirs(folder_path)
            print(f"Directory '{folder_path}' created successfully.")
        except Exception as e:
            print(f"Error removing directory {folder_path}: {e}")
    else:
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")


data_folder_check(SCREENSHOT_FOLDER_PATH)
data_folder_check(CAMERA_FOLDER_PATH)
data_folder_check(LIDAR_FOLDER_PATH)

# Connect to the BeamNG.drive simulator
beamng = BeamNGpy('localhost', 64256, home=BEAMNG_TECH_GAME_PATH_DIR)
beamng.close()
beamng.open()

time.sleep(2)

# Create a scenario
# scenario = Scenario('Daytona22', 'daytona22')
#
# # Spawn the ego vehicle
# ego_vehicle = Vehicle('ego_vehicle', model='moonhawk', licence='MSU-ATAV')
scenario = Scenario('east_coast_usa', 'tech_test', description='Random driving for research')
# Set up first vehicle, with two cameras, gforces sensor, lidar, electrical
# sensors, and damage sensors
ego_vehicle = Vehicle('ego_vehicle', model='etk800', license='RED', color='Red')

scenario.add_vehicle(ego_vehicle, pos=(-426.68, -43.59, 31.11), rot_quat=(0, 0, 1, 0))

scenario.make(beamng)

beamng.scenario.load(scenario)

# input message when ready to proceed with rest of code
print("Press Enter when ready to proceed with the rest of the code")
input()
print(beamng.get_gamestate())
print(beamng.get_current_vehicles())

ego_vehicle = beamng.get_current_vehicles()['ego_vehicle']
ego_vehicle.connect(beamng)

# electrics = Electrics()
# ego_vehicle.sensors.attach('electrics', electrics)
# beamng.scenario.start()
# beamng.ui.hide_hud()
# beamng.settings.set_deterministic(60)
# ego_vehicle.ai.set_mode('random')
# ego_vehicle.ai.drive_in_lane(True)

camera = Camera('camera1', beamng, ego_vehicle, is_render_instance=True,
                is_render_annotations=True, is_render_depth=True)
#
lidar = Lidar('lidar1', beamng, ego_vehicle)
ultrsonic = Ultrasonic('ultrasonic1', beamng, ego_vehicle)
print("Press Enter when ready to proceed with the rest of the code")
input()
steer_throttle = []
camera_sensor_data = []
lidar_sensor_data = []
ultrasonic_sensor_data = []
duration = 300
start_time = time.time()
while time.time() - start_time < duration:
    # Capture screenshot
    screenshot_path = '../screenshots/screenshot_{:03d}.png'.format(int(time.time()))
    camera_path = '../camera/camera_{:03d}.png'.format(int(time.time()))
    # rect = getRectAsImage(SCREENS[1])
    # rect.save(screenshot_path,format='png')

    # Record ego vehicle's control inputs (steering and throttle)
    # Get vehicle steering from beamngpy.sensors.Electrics
    ego_vehicle.sensors.poll()
    # camera_data = camera.poll()
    # lidar_data = lidar.poll()
    ultrsonic_data = ultrsonic.poll()
    print('Distance to obstacle: ', ultrsonic_data['distance'])
    # electrics_data = ego_vehicle.sensors['electrics']
    # print(electrics_data)
    # print(camera_data)
    # print(lidar_data)
    # color_image = camera_data['colour']
    # color_image.save(camera_path, format='PNG')
    # camera_sensor_data.append(camera_data)
    # lidar_sensor_data.append(lidar_data)
    # steer_throttle.append([electrics_data['steering'], electrics_data['throttle']])
    # Step the simulation
    # beamng.step(1)
    time.sleep(0.1)

# print("Time up for logging screenshots + steering + throttle input")
# # Save data along with timestamps or frame numbers
# with open('data.csv', 'w') as f:
#     f.write('time,steering,throttle\n')
#     for control in steer_throttle:
#         f.write('{},{},{}\n'.format(int(time.time()), control[0], control[1]))

# Cleanup
beamng.close()