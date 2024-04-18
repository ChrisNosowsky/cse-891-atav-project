import time
import shutil
import os
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Camera, Lidar, Damage, Timer, Ultrasonic, PowertrainSensor
from desktopmagic.screengrab_win32 import getDisplayRects, getRectAsImage

BEAMNG_GAME_PATH_DIR = "E:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"
BEAMNG_TECH_GAME_PATH_DIR = "E:/Program Files (x86)/Games/BeamNG.tech.v0.31.3.0"
SCREENSHOT_FOLDER_PATH = './screenshots/'
CAMERA_FOLDER_PATH = './camera/'
LIDAR_FOLDER_PATH = './lidar/'
SCREENS=(getDisplayRects())

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

beamng = BeamNGpy('localhost', 64256, home=BEAMNG_TECH_GAME_PATH_DIR)
beamng.close()
beamng.open()

time.sleep(2)

print("Press Enter when ready to proceed with the rest of the code\n")
print("This is the time to load up Freeroam mode on the game, select your car, select the replay to train on before proceeding.\n")
input()

print(beamng.get_gamestate())
print(beamng.get_current_vehicles())
print(beamng.get_levels())
ego_vehicle = next(iter(beamng.get_current_vehicles().values()))
ego_vehicle.connect(beamng)

beamng.settings.set_deterministic(60)
# ego_vehicle.ai.set_mode('random')

# camera = Camera('camera1', beamng, ego_vehicle, is_render_instance=True,
#                 is_render_annotations=True, is_render_depth=True)
lidar = Lidar('lidar1', beamng, ego_vehicle)

ultrasonic_forward = Ultrasonic('ultrasonic_forward', beamng, ego_vehicle, field_of_view_y=2, near_far_planes=(0.1, 20.1), range_direct_max_cutoff=20)

ultrasonic_left_30 = Ultrasonic('ultrasonic_left30', beamng, ego_vehicle, dir=(0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
ultrasonic_left_60 = Ultrasonic('ultrasonic_left60', beamng, ego_vehicle, dir=(0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)

ultrasonic_right_30 = Ultrasonic('ultrasonic_right30', beamng, ego_vehicle, dir=(-0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
ultrasonic_right_60 = Ultrasonic('ultrasonic_right60', beamng, ego_vehicle, dir=(-0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
# powertrain = PowertrainSensor('powertrain1', beamng, ego_vehicle)

electrics = Electrics()
damage = Damage()
timer = Timer()
ego_vehicle.sensors.attach('electrics', electrics)
# ego_vehicle.sensors.attach('damage', damage)

# ego_vehicle.sensors.attach('timer', timer)

print("Press Enter when ready to proceed with the rest of the code")
print("You are about to start training.")
input()

vehicle_data = []
timestamps = []
camera_sensor_data = []
lidar_sensor_data = []
ultrasonic_sensor_data = []
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

time.sleep(1)
start_time = time.time()
ego_vehicle.logging.start('data')
while time.time() - start_time < duration:
    current_epoch = int(time.time())
    timestamps.append(current_epoch)
    
    # screenshot_path = './screenshots/screenshot_{:03d}.png'.format(current_epoch)
    # camera_path = './camera/camera_{:03d}.png'.format(current_epoch)
    
    # rect = getRectAsImage(SCREENS[1])
    # rect.save(screenshot_path,format='png')
    
    ego_vehicle.sensors.poll()
    # camera_data = camera.poll()
    # lidar_data = lidar.poll()
    ultrasonic_forward_data = ultrasonic_forward.poll()
    
    ultrasonic_left_data_30 = ultrasonic_left_30.poll()
    ultrasonic_left_data_60 = ultrasonic_left_60.poll()
    
    ultrasonic_right_data_30 = ultrasonic_right_30.poll()
    ultrasonic_right_data_60 = ultrasonic_right_60.poll()
    # powertrain_data = powertrain.poll()
    
    electrics_data = ego_vehicle.sensors['electrics']
    # damage_data = ego_vehicle.sensors['damage']
    # timer_data = ego_vehicle.sensors['timer']
    
    # print("TIMER ", timer_data)
    # print("\n\nDAMAGE ", damage_data)
    print("\n\nELECTRICS ", electrics_data)
    # print("\n\nCAMERA DATA ", camera_data)
    # print("\n\nLIDAR DATA ", lidar_data['pointCloud'])
    # print("\n\nPOWERTRAIN DATA ", powertrain_data)
    print("\nULTRASONIC DATA FORWARD", ultrasonic_forward_data['distance'])
    print("\nULTRASONIC DATA LEFT 30", ultrasonic_left_data_30['distance'])
    print("\nULTRASONIC DATA LEFT 60", ultrasonic_left_data_60['distance'])
    print("\nULTRASONIC DATA RIGHT 30", ultrasonic_right_data_30['distance'])
    print("\nULTRASONIC DATA RIGHT 60", ultrasonic_right_data_60['distance'])
    
    
    # color_image = camera_data['colour']
    # color_image.save(camera_path, format='PNG')
    # camera_sensor_data.append(camera_data)
    # lidar_sensor_data.append(lidar_data)
    
    # vehicle_data.append([current_epoch, electrics_data['steering'], electrics_data['throttle'], 
    #                        electrics_data['brake'], electrics_data['gear'], lidar_data['pointCloud']])
    # Step the simulation
    # beamng.step(1)
    time.sleep(1)

print("Time up for logging screenshots + steering + throttle input")
# Save data along with timestamps or frame numbers
# with open('charlotte_roval.csv', 'w') as f:
#     f.write('uid,time,steering,throttle,brake,gear,lidar_pc\n')
#     for i, control in enumerate(vehicle_data):
#         f.write('{},{},{},{},{},{}\n'.format(control[0], control[1], control[2], control[3], control[4], control[5]))

# Cleanup
ego_vehicle.logging.stop()
beamng.close()


