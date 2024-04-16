import time
import shutil
import os
import csv
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Electrics, Camera, Lidar, Damage, Timer, Ultrasonic, PowertrainSensor, GPS, RoadsSensor
from desktopmagic.screengrab_win32 import getDisplayRects
from constants import *

SCREENS=(getDisplayRects())

class BeamNG:
    def __init__(self, hostname="localhost", port=64256, home=BEAMNG_TECH_GAME_PATH_DIR, verbose=False):
        self.beamng = BeamNGpy(hostname, port, home)
        self.verbose = verbose
        self.ego_vehicle = None
        self.vehicle_sensors = []
        self.vehicle_data = []
        self.vehicle_data_dict = {}
        self.cur_throttle = 0
        self.cur_steering = 0
        self.cur_brake = 0
        self.cur_gear = 0
        self.cur_rpm = 0
        self.cur_track_pos = 0
        self.cur_angle = 0
        self.speed_x = 0
        self.speed_y = 0
        self.speed_z = 0
        self.cur_track_dist_forward = 0
        self.cur_track_dist_right_30 = 0
        self.cur_track_dist_right_60 = 0
        self.cur_track_dist_left_30 = 0
        self.cur_track_dist_left_60 = 0
        self.road_geometry = None
        self.road_width = 10
        self.vehicle_pos = (43.951, 127.815, 180.100)
        self.vehicle_rot_quat = (0, 0, 1, 1)
    
    @staticmethod
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

    def is_within_track_border(self, road_geometry, cur_gps_pos):
        track_pos = 15
        cur_x, cur_y = cur_gps_pos
        is_within = False
        
        left_edge_x = np.array([e['left'][0] for e in road_geometry])
        left_edge_y = np.array([e['left'][1] for e in road_geometry])
        right_edge_x = np.array([e['right'][0] for e in road_geometry])
        right_edge_y = np.array([e['right'][1] for e in road_geometry])
        middles_x = np.array([e['middle'][0] for e in road_geometry])
        middles_y = np.array([e['middle'][1] for e in road_geometry])
        for i in range(len(road_geometry) - 1): # For each node, get left and right edges
            segment_min_x = min(left_edge_x[i], right_edge_x[i], left_edge_x[i+1], right_edge_x[i+1])
            segment_max_x = max(left_edge_x[i], right_edge_x[i], left_edge_x[i+1], right_edge_x[i+1])
            segment_min_y = min(left_edge_y[i], right_edge_y[i], left_edge_y[i+1], right_edge_y[i+1])
            segment_max_y = max(left_edge_y[i], right_edge_y[i], left_edge_y[i+1], right_edge_y[i+1])
            segment_min_middle_x = min(middles_x[i], middles_x[i+1])
            segment_max_middle_x = max(middles_x[i], middles_x[i+1])
            segment_max_middle_y = max(middles_y[i], middles_y[i+1])
            segment_min_middle_y = min(middles_y[i], middles_y[i+1])
            if segment_min_x <= cur_x <= segment_max_x and segment_min_y <= cur_y <= segment_max_y:
                cur_pos = cur_x + cur_y
                mid_pos = np.mean([segment_min_middle_x, segment_max_middle_x]) + np.mean([segment_min_middle_y, segment_max_middle_y])
                track_pos = np.abs(cur_pos - mid_pos) / self.road_width
                is_within = True
                break
        print("IS WITHIN? ", is_within)
        return is_within, track_pos


    def configure_beamng(self):
        self.beamng.settings.set_deterministic(60)
        
    def close_beamng(self):
        self.beamng.close()
    
    def recover_vehicle(self):
        self.ego_vehicle.teleport(pos=self.vehicle_pos, rot_quat=self.vehicle_rot_quat)
    
    def get_current_actions(self):
        return {"throttle": self.cur_throttle, "steering": self.cur_steering, "brake": self.cur_brake}
    
    def apply_actions(self, actions):
        print("APPLYING THE CONTROL!!: ", actions)
        self.ego_vehicle.control(throttle=actions["throttle"], steering=actions["steering"], brake=actions["brake"])
    
    def poll_sensors(self):
        camera: Camera = self.vehicle_sensors[0]
        lidar: Lidar = self.vehicle_sensors[1]
        gps: GPS = self.vehicle_sensors[2]
        road_sensor: RoadsSensor = self.vehicle_sensors[3]
        powertrain: PowertrainSensor = self.vehicle_sensors[4]
        ultrasonic_forward: Ultrasonic = self.vehicle_sensors[5]
        ultrasonic_left_30: Ultrasonic = self.vehicle_sensors[6]
        ultrasonic_left_60: Ultrasonic = self.vehicle_sensors[7]
        ultrasonic_right_30: Ultrasonic = self.vehicle_sensors[8]
        ultrasonic_right_60: Ultrasonic = self.vehicle_sensors[9]
        
        self.ego_vehicle.sensors.poll()
        camera_data = camera.poll()
        lidar_data = lidar.poll()
        gps_data = gps.poll()
        road_data = road_sensor.poll()
        powertrain_data = powertrain.poll()
        ultrasonic_forward_data = ultrasonic_forward.poll()
        ultrasonic_left_data_30 = ultrasonic_left_30.poll()
        ultrasonic_left_data_60 = ultrasonic_left_60.poll()
        ultrasonic_right_data_30 = ultrasonic_right_30.poll()
        ultrasonic_right_data_60 = ultrasonic_right_60.poll()
        
        electrics_data = self.ego_vehicle.sensors['electrics']
        damage_data = self.ego_vehicle.sensors['damage']
        
        self.cur_brake = electrics_data['brake']
        if (electrics_data['gear'] == 'N'):
            self.cur_gear = 0
        elif (electrics_data['gear'] == 'R'):
            self.cur_gear = -1
        elif (electrics_data['gear'] == 'D'):
            self.cur_gear = 1
        else:
            self.cur_gear = electrics_data['gear']
        self.cur_rpm = electrics_data['rpm']
        try:
            self.cur_angle = road_data[0]['headingAngle']
        except:
            print("Wait")
            time.sleep(1)
            self.cur_angle = 0
        is_within, track_pos = self.is_within_track_border(self.road_geometry, (gps_data[0]['x'], gps_data[0]['y']))
        if not is_within:
            print("Out of track border")
            track_pos = 15
        
        self.cur_track_pos = track_pos
        self.speed_x = electrics_data['accXSmooth']
        self.speed_y = electrics_data['accYSmooth']
        self.speed_z = electrics_data['accZSmooth']
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
            "angle": self.cur_angle,
            "track_pos": self.cur_track_pos,
            "speed_x": self.speed_x,
            "speed_y": self.speed_y,
            "speed_z": self.speed_z,
            "engine_temp": self.cur_engine_temp,
            "wheelspin": self.cur_wheelspin,
            "damage": self.cur_damage,
            "track_dist_forward": self.cur_track_dist_forward,
            "track_dist_right_30": self.cur_track_dist_right_30,
            "track_dist_right_60": self.cur_track_dist_right_60,
            "track_dist_left_30": self.cur_track_dist_left_30,
            "track_dist_left_60": self.cur_track_dist_left_60
        }
        
        current_epoch = int(time.time())
        self.vehicle_data.append([current_epoch, electrics_data['steering'], electrics_data['throttle'], 
                    electrics_data['brake'], electrics_data['gear'], lidar_data['pointCloud']])
        
        if self.verbose:
            # print("TIMER ", timer_data)
            # print("\n\nDAMAGE ", damage_data)
            print("\n\nELECTRICS ", electrics_data)
            # print("\n\nULTRASONIC DATA FORWARD", ultrasonic_forward_data['distance'])
            # print("\n\nULTRASONIC DATA LEFT 30", ultrasonic_left_data_30['distance'])
            # print("\n\nULTRASONIC DATA LEFT 60", ultrasonic_left_data_60['distance'])
            # print("\n\nULTRASONIC DATA RIGHT 30", ultrasonic_right_data_30['distance'])
            # print("\n\nULTRASONIC DATA RIGHT 60", ultrasonic_right_data_60['distance'])
            # print("\n\nCAMERA DATA ", camera_data)
            # print("\n\nLIDAR DATA ", lidar_data)
            # print("\n\nPOWERTRAIN DATA ", powertrain_data)
        
        return self.vehicle_data_dict
        
        
    def write_logged_data(self, filename):
        with open(filename, 'w') as f:
            f.write('uid,time,steering,throttle,brake,gear,lidar_pc\n')
            for i, control in enumerate(self.vehicle_data):
                f.write('{},{},{},{},{},{}\n'.format(control[0], control[1], control[2], control[3], control[4], control[5]))

        
    def run_simulator(self):
        self.beamng.close()
        self.beamng.open()

        time.sleep(2)
        scenario = Scenario('north wilkesboro', 'tech_test', description='Random driving for research')
        self.ego_vehicle = Vehicle('ego_vehicle', model='pickup', license='RED', color='Red')


        scenario.add_vehicle(self.ego_vehicle, pos=self.vehicle_pos, rot_quat=self.vehicle_rot_quat)
        road_a = Road('track_rubber', rid='nwb_oval_road')

        with open("data/north_wilkensboro_nodes.csv", mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                node = tuple(float(val) if i != 3 else int(val) for i, val in enumerate(row))
                print(node)
                road_a.add_nodes(node)

        scenario.add_road(road_a)

        scenario.make(self.beamng)
        self.beamng.scenario.load(scenario)
        self.beamng.scenario.start()
        
        print("Press Enter when ready to proceed with the rest of the code\n")
        print("This is the time to load up Freeroam mode on the game, select your car, select the replay to train on before proceeding.\n")
        input()
        
        self.ego_vehicle.connect(self.beamng)
        self.beamng.settings.set_deterministic(60)
        
        camera = Camera('camera1', self.beamng, self.ego_vehicle, is_render_instance=True,
                        is_render_annotations=True, is_render_depth=True)
        lidar = Lidar('lidar1', self.beamng, self.ego_vehicle)
        road_sensor = RoadsSensor('road', bng=self.beamng, vehicle=self.ego_vehicle)
        gps = GPS('gps', bng=self.beamng, vehicle=self.ego_vehicle)
        
        ultrasonic_forward = Ultrasonic('ultrasonic_forward', self.beamng, self.ego_vehicle, field_of_view_y=2, near_far_planes=(0.1, 20.1), range_direct_max_cutoff=20)

        ultrasonic_left_30 = Ultrasonic('ultrasonic_left30', self.beamng, self.ego_vehicle, dir=(0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
        ultrasonic_left_60 = Ultrasonic('ultrasonic_left60', self.beamng, self.ego_vehicle, dir=(0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)

        ultrasonic_right_30 = Ultrasonic('ultrasonic_right30', self.beamng, self.ego_vehicle, dir=(-0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
        ultrasonic_right_60 = Ultrasonic('ultrasonic_right60', self.beamng, self.ego_vehicle, dir=(-0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
                
        powertrain = PowertrainSensor('powertrain1', self.beamng, self.ego_vehicle)
        
        electrics = Electrics()
        damage = Damage()
        timer = Timer()
        self.ego_vehicle.sensors.attach('electrics', electrics)
        self.ego_vehicle.sensors.attach('damage', damage)
        
        self.vehicle_sensors = [camera, lidar, gps, road_sensor, powertrain, ultrasonic_forward, ultrasonic_left_30, ultrasonic_left_60, ultrasonic_right_30, ultrasonic_right_60]
        
        print("Press Enter when ready to proceed with the rest of the code")
        print("You are about to start training.")
        input()

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
        
        self.road_geometry = self.beamng.scenario.get_road_edges('nwb_oval_road')
        
        time.sleep(1)
        print("Game environment is ready.")
        
if __name__ == '__main__':
    # Testing individual class below
    
    beamng = BeamNG(verbose=True)
    
    beamng.data_folder_check(SCREENSHOT_FOLDER_PATH)
    beamng.data_folder_check(CAMERA_FOLDER_PATH)
    beamng.data_folder_check(LIDAR_FOLDER_PATH)
    
    beamng.run_simulator()
    
    duration = 600
    start_time = time.time()
        
    while time.time() - start_time < duration:
        beamng.poll_sensors()
        time.sleep(1)
    
    print("Times up. Closing BeamNG")
    beamng.close_beamng()
    
    beamng.write_logged_data('north_wilkesboro_data.csv')