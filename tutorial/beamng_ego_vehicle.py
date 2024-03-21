import time
import pyautogui
import shutil
import os
from PIL import ImageGrab
from functools import partial
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics

STEAM_GAME_PATH_DIR = "E:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"
ImageGrab.grab = partial(ImageGrab.grab, all_screens=True)

folder_path = '../screenshots/'

if os.path.exists(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Removed directory {folder_path}")
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    except Exception as e:
        print(f"Error removing directory {folder_path}: {e}")


# Connect to the BeamNG.drive simulator
beamng = BeamNGpy('localhost', 64256, home=STEAM_GAME_PATH_DIR)
beamng.open(launch=True)

# Create a scenario
scenario = Scenario('east_coast_usa', 'freeroam')

# Spawn the ego vehicle
ego_vehicle = Vehicle('ego_vehicle', model='pickup', licence='PYTHON')
scenario.add_vehicle(ego_vehicle, pos=(-426.68, -43.59, 31.11), rot_quat=(0, 0, 1, 0))
ego_vehicle.attach_sensor('electrics', Electrics())

scenario.make(beamng)

beamng.scenario.load(scenario)
beamng.scenario.start()

steer_throttle = []
monitor_x, monitor_y = 0, 0
monitor_width, monitor_height = 1920, 1080
region = (monitor_x, monitor_y, monitor_width, monitor_height)


duration = 60
start_time = time.time()
while time.time() - start_time < duration:

    # Capture screenshot
    screenshot_path = '../screenshots/screenshot_{:03d}.png'.format(int(time.time()))
    screenshot = pyautogui.screenshot(region=region)
    
    screenshot.save(screenshot_path)

    # Record ego vehicle's control inputs (steering and throttle)
    # Get vehicle steering from beamngpy.sensors.Electrics
    electrics_data = ego_vehicle.sensors['electrics'].data
    print(electrics_data)
    # steering = ego_vehicle.logger.
    # throttle = ego_vehicle.state['throttle_input']
    # steer_throttle.append([steering, throttle])

    # Step the simulation
    beamng.step(1)

print("Time up for logging screenshots + steering + throttle input")
# Save data along with timestamps or frame numbers
with open('data.csv', 'w') as f:
    f.write('time,steering,throttle\n')
    for control in steer_throttle:
        f.write('{},{},{}\n'.format(time.time(), control[0], control[1]))

# Cleanup
beamng.close()