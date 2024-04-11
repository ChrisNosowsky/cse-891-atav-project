import csv
import time
import shutil
import os
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging, Road
from beamngpy.sensors import Electrics, Camera, Lidar, Damage, Timer, Ultrasonic, PowertrainSensor, GPS, RoadsSensor
from desktopmagic.screengrab_win32 import getDisplayRects, getRectAsImage

BEAMNG_GAME_PATH_DIR = "E:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"
BEAMNG_TECH_GAME_PATH_DIR = "E:/Program Files (x86)/Games/BeamNG.tech.v0.31.3.0"
MISHA_BEAMNG_TECH_GAME_PATH_DIR = "C:/Users/laptop/Desktop/BeamNG.tech.v0.31.3.0"
SCREENSHOT_FOLDER_PATH = './screenshots/'
CAMERA_FOLDER_PATH = './camera/'
LIDAR_FOLDER_PATH = './lidar/'
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

set_up_simple_logging()
beamng = BeamNGpy('localhost', 64256, home=MISHA_BEAMNG_TECH_GAME_PATH_DIR)
# beamng.close()
beamng.open()

time.sleep(2)
scenario = Scenario('east_coast_usa', 'tech_test', description='Random driving for research')
    # Set up first vehicle, with two cameras, gforces sensor, lidar, electrical
    # sensors, and damage sensors
vehicle = Vehicle('ego_vehicle', model='etk800', license='RED', color='Red')
electrics = Electrics()
damage = Damage()
#gps = GPS('gps', bng=beamng, vehicle=vehicle)
print(scenario._get_roads_list())
# roadsensor = RoadsSensor('road', bng=beamng, vehicle=vehicle)
vehicle.attach_sensor('electrics', electrics)
vehicle.attach_sensor('damage', damage)
# vehicle.attach_sensor('gps', gps)
scenario.add_vehicle(vehicle, pos=(-426.68, -43.59, 31.11), rot_quat=(0, 0, 1, 0))

    # Compile the scenario and place it in BeamNG's map folder
scenario.make(beamng)
beamng.scenario.load(scenario)
beamng.scenario.start()
print("Press Enter when ready to proceed with the rest of the code\n")
print(
    "This is the time to load up Freeroam mode on the game, select your car, select the replay to train on before proceeding.\n")
input()

print(beamng.get_gamestate())
print(beamng.get_current_vehicles())

ego_vehicle = next(iter(beamng.get_current_vehicles().values()))
ego_vehicle.connect(beamng)

beamng.settings.set_deterministic(60)

print("Press Enter when ready to proceed with the rest of the code")
print("You are about to start training.")
input()
roads = scenario._get_roads_list()
roads_2 = scenario._get_mesh_roads_list()
vehicle_data = []
timestamps = []
camera_sensor_data = []
lidar_sensor_data = []
ultrasonic_sensor_data = []
gps_data = []
duration = 60

beamng.control.pause()
paused = True
countdown = 3

while paused:
    if countdown == 1:
        paused = False
    print("Game is paused. Unpausing to train in {}...".format(countdown))
    countdown -= 1
    time.sleep(1)

beamng.control.resume()

time.sleep(1)
start_time = time.time()
vehicle.logging.start('data')
results = []
while time.time() - start_time < duration:
    current_epoch = int(time.time())
    timestamps.append(current_epoch)

    vehicle.sensors.poll()

    electrics_data = vehicle.sensors['electrics']
    damage_data = vehicle.sensors['damage']
    # gps_data = vehicle.sensors['gps']
    #
    # print("GPS: ", gps_data)


    # print("\n\nELECTRICS ", electrics_data)
    # print("\nWheel spin velocity: ", electrics_data['wheelspeed'])
    # print("\nThrottle Input: ", electrics_data['throttle_input'])
    # print("\nSpeed of the Vehicle: ", electrics_data['airspeed'])
    # print("\nBrakes: ", electrics_data['brake_input'])
    # print("\nRPM: ", electrics_data['rpm'])
    # print("\nSteering: ", electrics_data['steering_input'])
    # print("\nOil Temperature: ", electrics_data['oil_temperature'])
    # print("\Water Temperature: ", electrics_data['water_temperature'])
    # - gear (int):
    # - gear_a (int): Gear selected in automatic mode.
    # - gear_index (int):
    # - gear_m (int): Gear selected in manual mode.
    # print("\nGear Mode Index: ", electrics_data['gearModeIndex'])

    # print("\nVelocity  of the car on X axis: ", electrics_data['accXSmooth'])
    # print("\nVelocity  of the car on Y axis: ", electrics_data['accYSmooth'])
    # print("\nVelocity  of the car on Z axis: ", electrics_data['accZSmooth'])

    # print("\n\nDAMAGE ", damage_data['damage']
    results.append((electrics_data['wheelspeed'], electrics_data['throttle_input'], electrics_data['airspeed'], electrics_data['brake_input'], electrics_data['rpm'], electrics_data['steering_input'], electrics_data['oil_temperature'], electrics_data['water_temperature'], electrics_data['gearModeIndex'], electrics_data['accXSmooth'], electrics_data['accYSmooth'], electrics_data['accZSmooth'], damage_data['damage']))


print("Time up for logging screenshots + steering + throttle input")
# Save data along with timestamps or frame numbers
# with open('charlotte_roval.csv', 'w') as f:
#     f.write('uid,time,steering,throttle,brake,gear,lidar_pc\n')
#     for i, control in enumerate(vehicle_data):
#         f.write('{},{},{},{},{},{}\n'.format(control[0], control[1], control[2], control[3], control[4], control[5]))
csv_file_path = "../cse-891-atav-project/data/results.csv"
with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Wheel spin velocity", "Throttle Input", "Speed of the Vehicle", "Brakes", "RPM", "Steering",
                         "Oil Temperature", "Water Temperature", "Gear", "Velocity  of the car on X axis",
                         "Velocity  of the car on Y axis", "Velocity  of the car on z axis", "Damage"])
    csv_writer.writerow(results)
time.sleep(1)
# Cleanup
ego_vehicle.logging.stop()
beamng.close()


