import csv
import time
import shutil
import os
import numpy as np
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


def is_within_track_border(road_geometry, cur_gps_pos):
    cur_x, cur_y = cur_gps_pos
    is_within = False
    
    left_edge_x = np.array([e['left'][0] for e in road_geometry])
    left_edge_y = np.array([e['left'][1] for e in road_geometry])
    right_edge_x = np.array([e['right'][0] for e in road_geometry])
    right_edge_y = np.array([e['right'][1] for e in road_geometry])
    for i in range(len(road_geometry) - 1):
        segment_min_x = min(left_edge_x[i], right_edge_x[i], left_edge_x[i+1], right_edge_x[i+1])
        segment_max_x = max(left_edge_x[i], right_edge_x[i], left_edge_x[i+1], right_edge_x[i+1])
        segment_min_y = min(left_edge_y[i], right_edge_y[i], left_edge_y[i+1], right_edge_y[i+1])
        segment_max_y = max(left_edge_y[i], right_edge_y[i], left_edge_y[i+1], right_edge_y[i+1])

        if segment_min_x <= cur_x <= segment_max_x and segment_min_y <= cur_y <= segment_max_y:
            is_within = True
            break
    print("IS WITHIN? ", is_within)
    return is_within
    

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
beamng = BeamNGpy('localhost', 64256, home=BEAMNG_TECH_GAME_PATH_DIR)
# beamng.close()
beamng.open()

time.sleep(2)
scenario = Scenario('north wilkesboro', 'tech_test', description='Random driving for research')
    # Set up first vehicle, with two cameras, gforces sensor, lidar, electrical
    # sensors, and damage sensors
vehicle = Vehicle('ego_vehicle', model='pickup', license='RED', color='Red')
electrics = Electrics()
damage = Damage()
vehicle.attach_sensor('electrics', electrics)
vehicle.attach_sensor('damage', damage)
scenario.add_vehicle(vehicle, pos=(43.951, 127.815, 180.100), rot_quat=(0, 0, 1, 1))
road_a = Road('track_rubber', rid='nwb_oval_road')

with open("data/north_wilkensboro_nodes.csv", mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        node = tuple(float(val) if i != 3 else int(val) for i, val in enumerate(row))
        print(node)
        road_a.add_nodes(node)

scenario.add_road(road_a)

print(scenario.roads)
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
print(beamng.get_levels())

ego_vehicle = next(iter(beamng.get_current_vehicles().values()))
ego_vehicle.connect(beamng)
ego_vehicle.ai_set_script
beamng.settings.set_deterministic(60)
road_sensor = RoadsSensor('road', bng=beamng, vehicle=vehicle)
gps = GPS('gps', bng=beamng, vehicle=vehicle)
print("Press Enter when ready to proceed with the rest of the code")
print("You are about to start training.")
input()

vehicle_data = []
timestamps = []
camera_sensor_data = []
lidar_sensor_data = []
ultrasonic_sensor_data = []
gps_data = []
duration = 600

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

road_geometry = beamng.scenario.get_road_edges('nwb_oval_road')



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
    gps_data = gps.poll()
    road_sensor_data = road_sensor.poll()
    
    # print("GPS: ", gps_data[0])
    is_within_track_border(road_geometry, (gps_data[0]['x'], gps_data[0]['y']))
    
    print("Velocity X: ", electrics_data['accXSmooth'])
    print("Velocity Y: ", electrics_data['accYSmooth'])
    print("Velocity Z: ", electrics_data['accZSmooth'])
    print("Heading Angle: ", road_sensor_data[0]['headingAngle'])
    
    time.sleep(5)

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


