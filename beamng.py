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

class BeamNG:
    def __init__(self, hostname="localhost", port=64256, home=BEAMNG_TECH_GAME_PATH_DIR):
        self.beamng = BeamNGpy(hostname, port, home)
        self.ego_vehicle = None
        self.vehicle_data = []
        self.vehicle_data_dict = {}
        self.cur_throttle = 0
        self.cur_steering = 0
        self.cur_brake = 0
        self.cur_gear = 0
        self.cur_rpm = 0
        self.speed_x = 0
        self.speed_y = 0
        self.cur_track_dist_forward = 0
        self.cur_track_dist_right_30 = 0
        self.cur_track_dist_right_60 = 0
        self.cur_track_dist_left_30 = 0
        self.cur_track_dist_left_60 = 0
    
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

    def configure_beamng(self):
        self.beamng.settings.set_deterministic(60)
        
    def close_beamng(self):
        self.beamng.close()
    
    def get_current_actions(self):
        return {"throttle": self.cur_throttle, "steering": self.cur_steering, "brake": self.cur_brake}
    
    def apply_actions(self, actions):
        self.ego_vehicle.control(throttle=actions["throttle"], steering=actions["steering"], brake=actions["brake"])
    
    def poll_sensors(self, camera: Camera, 
                     lidar: Lidar, 
                     powertrain: PowertrainSensor,
                     ultrasonic_forward: Ultrasonic,
                     ultrasonic_left_30: Ultrasonic,
                     ultrasonic_left_60: Ultrasonic,
                     ultrasonic_right_30: Ultrasonic,
                     ultrasonic_right_60: Ultrasonic):
        self.ego_vehicle.sensors.poll()
        
        camera_data = camera.poll()
        lidar_data = lidar.poll()
        powertrain_data = powertrain.poll()
        ultrasonic_forward_data = ultrasonic_forward.poll()
        ultrasonic_left_data_30 = ultrasonic_left_30.poll()
        ultrasonic_left_data_60 = ultrasonic_left_60.poll()
        ultrasonic_right_data_30 = ultrasonic_right_30.poll()
        ultrasonic_right_data_60 = ultrasonic_right_60.poll()
        
        electrics_data = self.ego_vehicle.sensors['electrics']
        damage_data = self.ego_vehicle.sensors['damage']
        
        self.cur_brake = electrics_data['brake']
        self.cur_gear = electrics_data['gear']
        self.cur_rpm = electrics_data['rpm']
        self.speed_x = electrics_data['accXSmooth']
        self.speed_y = electrics_data['accYSmooth']
        self.cur_throttle = electrics_data['throttle']
        self.cur_steering = electrics_data['steering']
        self.cur_engine_temp = electrics_data['water_temperature']
        self.cur_wheelspin = electrics_data['wheelspeed']
        self.cur_damage = damage_data['damage']
        self.cur_track_dist_forward = ultrasonic_forward_data['distance']
        self.cur_track_dist_right_30 = ultrasonic_right_data_30['distance']
        self.cur_track_dist_right_60 = ultrasonic_right_data_60['distance']
        self.cur_track_dist_left_30 = ultrasonic_left_data_30['distance']
        self.cur_track_dist_left_60 = ultrasonic_left_data_60['distance']
        
        self.vehicle_data_dict = {
            "gear": self.cur_gear,
            "rpm": self.cur_rpm,
            "speed_x": self.speed_x,
            "speed_y": self.speed_y,
            "engine_temp": self.cur_engine_temp,
            "wheelspin": self.cur_wheelspin,
            "damage": self.cur_damage,
            "track_dist_forward": self.cur_track_dist_forward,
            "track_dist_right_30": self.cur_track_dist_right_30,
            "track_dist_right_60": self.cur_track_dist_right_60,
            "track_dist_left_30": self.cur_track_dist_left_30,
            "track_dist_left_60": self.cur_track_dist_left_60
        }
        
        return self.vehicle_data_dict
        
        
    def write_logged_data(self, filename):
        with open(filename, 'w') as f:
            f.write('uid,time,steering,throttle,brake,gear,lidar_pc\n')
            for i, control in enumerate(self.vehicle_data):
                f.write('{},{},{},{},{},{}\n'.format(control[0], control[1], control[2], control[3], control[4], control[5]))

        
    def run_simulator(self):
        print("Press Enter when ready to proceed with the rest of the code\n")
        print("This is the time to load up Freeroam mode on the game, select your car, select the replay to train on before proceeding.\n")
        input()

        print(self.beamng.get_gamestate())
        print(self.beamng.get_current_vehicles())
        
        ego_vehicle = next(iter(self.beamng.get_current_vehicles().values()))
        ego_vehicle.connect(self.beamng)
        self.beamng.settings.set_deterministic(60)

        camera = Camera('camera1', self.beamng, ego_vehicle, is_render_instance=True,
                        is_render_annotations=True, is_render_depth=True)
        lidar = Lidar('lidar1', self.beamng, ego_vehicle)
        
        ultrasonic_forward = Ultrasonic('ultrasonic_forward', beamng, ego_vehicle, field_of_view_y=2, near_far_planes=(0.1, 20.1), range_direct_max_cutoff=20)

        ultrasonic_left_30 = Ultrasonic('ultrasonic_left30', beamng, ego_vehicle, dir=(0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
        ultrasonic_left_60 = Ultrasonic('ultrasonic_left60', beamng, ego_vehicle, dir=(0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)

        ultrasonic_right_30 = Ultrasonic('ultrasonic_right30', beamng, ego_vehicle, dir=(-0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
        ultrasonic_right_60 = Ultrasonic('ultrasonic_right60', beamng, ego_vehicle, dir=(-0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
                
        powertrain = PowertrainSensor('powertrain1', self.beamng, ego_vehicle)
        electrics = Electrics()
        damage = Damage()
        timer = Timer()
        ego_vehicle.sensors.attach('electrics', electrics)
        ego_vehicle.sensors.attach('damage', damage)
        ego_vehicle.control()
        print("Press Enter when ready to proceed with the rest of the code")
        print("You are about to start training.")
        input()
        
        timestamps = []
        camera_sensor_data = []
        lidar_sensor_data = []
        duration = 60

        self.beamng.control.pause()
        paused = True
        countdown = 3
        
        while paused:
            if countdown == 1:
                paused = False
            print("Game is paused. Unpausing to train in {}...".format(countdown))
            countdown -= 1
            time.sleep(1)
            
        self.beamng.control.resume()
        
        time.sleep(1)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            current_epoch = int(time.time())
            timestamps.append(current_epoch)
            
            # screenshot_path = './screenshots/screenshot_{:03d}.png'.format(current_epoch)
            # camera_path = './camera/camera_{:03d}.png'.format(current_epoch)
            
            # rect = getRectAsImage(SCREENS[1])
            # rect.save(screenshot_path,format='png')
            
            ego_vehicle.sensors.poll()
            camera_data = camera.poll()
            lidar_data = lidar.poll()
            powertrain_data = powertrain.poll()
            
            electrics_data = ego_vehicle.sensors['electrics']
            damage_data = ego_vehicle.sensors['damage']
            # timer_data = ego_vehicle.sensors['timer']
            
            # print("TIMER ", timer_data)
            print("\n\nDAMAGE ", damage_data)
            print("\n\nELECTRICS ", electrics_data)
            # print("\n\nCAMERA DATA ", camera_data)
            # print("\n\nLIDAR DATA ", lidar_data)
            print("\n\nPOWERTRAIN DATA ", powertrain_data)
            
            # color_image = camera_data['colour']
            # color_image.save(camera_path, format='PNG')
            # camera_sensor_data.append(camera_data)
            # lidar_sensor_data.append(lidar_data)
            
            self.vehicle_data.append([current_epoch, electrics_data['steering'], electrics_data['throttle'], 
                                electrics_data['brake'], electrics_data['gear'], lidar_data['pointCloud']])
            # Step the simulation
            # beamng.step(1)
            # time.sleep(0.1)
        print("Time up for logging screenshots + steering + throttle input")
        
        
if __name__ == '__main__':
    beamng = BeamNG()
    
    beamng.data_folder_check(SCREENSHOT_FOLDER_PATH)
    beamng.data_folder_check(CAMERA_FOLDER_PATH)
    beamng.data_folder_check(LIDAR_FOLDER_PATH)
    
    beamng.run_simulator()
    beamng.close_beamng()
    
    beamng.write_logged_data('charlotte_roval.csv')