import time
import shutil
import os
import csv
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import Electrics, Damage, Timer, Ultrasonic, GPS, RoadsSensor
from desktopmagic.screengrab_win32 import getDisplayRects
from constants import *

SCREENS=(getDisplayRects())

class BeamNG:
    def __init__(self, hostname="localhost", port=64256, 
                 home=BEAMNG_TECH_GAME_PATH_DIR, vehicle="pickup", 
                 track="north wilkesboro", road_width=10, verbose=False):
        self.beamng = BeamNGpy(hostname, port, home)
        self.verbose = verbose
        self.vehicle = vehicle
        self.track = track
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
        self.road_width = road_width
        self.vehicle_pos = (43.951, 127.815, 180.100)
        self.vehicle_rot_quat = (0, 0, 1, 1)
        self.previous_distance = 0
    
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


    @staticmethod
    def point_to_line_distance(px, py, x1, y1, x2, y2):
        # Calculate the line vector and point vector
        line_vec = np.array([x2 - x1, y2 - y1])
        pnt_vec = np.array([px - x1, py - y1])

        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        pnt_vec_scaled = pnt_vec / line_len

        # Project point onto the line (dot product)
        t = np.dot(line_unitvec, pnt_vec_scaled)    
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

        nearest = line_vec * t
        dist = np.linalg.norm(pnt_vec - nearest)

        # Determine side (cross product)
        side = np.cross(line_unitvec, pnt_vec)
        side_sign = np.sign(side)

        return dist * side_sign  # Returns negative if point is left of the line 


    def get_car_position_relative_to_road(self, middle_x, middle_y, cur_x, cur_y, road_width):
        min_dist = float('inf')
        side = 0
        for i in range(len(middle_x) - 1):
            dist = self.point_to_line_distance(cur_x, cur_y, middle_x[i], middle_y[i], middle_x[i+1], middle_y[i+1])
            if abs(dist) < abs(min_dist):
                min_dist = dist
                side = np.sign(dist)
        
        normalized_dist = min_dist / (road_width / 2)  # Normalizing by half the road width
        return normalized_dist, side


    def update_car_heading_and_position(self, cur_x, cur_y, middle_x, middle_y, road_width):
        normalized_dist, current_side = self.get_car_position_relative_to_road(middle_x, middle_y, cur_x, cur_y, road_width)
        normalized_dist, current_side = -normalized_dist, -current_side
        moving_direction = UNKNOWN
        
        abs_normalized_dist = abs(normalized_dist)
        # print("ABS Normalized distance: ", normalized_dist, "Current side: ", "Left" if current_side < 0 else "Right")
        
        if self.previous_distance != 0:  # Check if there is a previous distance for comparison
            if abs_normalized_dist > abs(self.previous_distance):    # Away from center line
                if current_side < 0:
                    moving_direction = TURNING_LEFT
                else:
                    moving_direction = TURNING_RIGHT
            elif abs_normalized_dist < abs(self.previous_distance):
                if current_side < 0:
                    moving_direction = TURNING_RIGHT
                else:
                    moving_direction = TURNING_LEFT
        # Update the previous states for next computation
        self.previous_distance = abs_normalized_dist

        return normalized_dist, current_side, moving_direction


    def is_within_track_border(self, road_geometry, cur_gps_pos):
        track_pos = 5
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
            if segment_min_x <= cur_x <= segment_max_x and segment_min_y <= cur_y <= segment_max_y:
                is_within = True
                break
        
        track_pos, side, moving_direction = self.update_car_heading_and_position(cur_x, cur_y, middles_x, middles_y, self.road_width)
        # track_pos [-1, 1] --> 0 Center; (0, 1] Right; [-1, 0) Left
        # side == -1 is left, side == +1 is right
        # print("IS WITHIN? ", is_within)
        
        return is_within, track_pos, moving_direction


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
        gps: GPS = self.vehicle_sensors[0]
        road_sensor: RoadsSensor = self.vehicle_sensors[1]
        ultrasonic_forward: Ultrasonic = self.vehicle_sensors[2]
        ultrasonic_left_30: Ultrasonic = self.vehicle_sensors[3]
        ultrasonic_left_60: Ultrasonic = self.vehicle_sensors[4]
        ultrasonic_right_30: Ultrasonic = self.vehicle_sensors[5]
        ultrasonic_right_60: Ultrasonic = self.vehicle_sensors[6]
        
        self.ego_vehicle.sensors.poll()
        gps_data = gps.poll()
        road_data = road_sensor.poll()
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
        is_within, track_pos, car_heading = self.is_within_track_border(self.road_geometry, (gps_data[0]['x'], gps_data[0]['y']))
        
        try:
            self.cur_angle = road_data[0]['headingAngle']
            if car_heading == TURNING_LEFT:
                self.cur_angle = -self.cur_angle
        except:
            print("Wait")
            time.sleep(1)
            self.cur_angle = 0
        
        self.cur_track_pos = track_pos
        self.speed_x = self.ego_vehicle.state['vel'][0]
        self.speed_y = self.ego_vehicle.state['vel'][1]
        self.speed_z = self.ego_vehicle.state['vel'][2]
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
        
        if self.verbose:
            # print("TIMER ", timer_data)
            print("\n\nDAMAGE ", damage_data)
            print("\n\nELECTRICS ", electrics_data)
            print("\n\nULTRASONIC DATA FORWARD", ultrasonic_forward_data['distance'])
            print("\n\nULTRASONIC DATA LEFT 30", ultrasonic_left_data_30['distance'])
            print("\n\nULTRASONIC DATA LEFT 60", ultrasonic_left_data_60['distance'])
            print("\n\nULTRASONIC DATA RIGHT 30", ultrasonic_right_data_30['distance'])
            print("\n\nULTRASONIC DATA RIGHT 60", ultrasonic_right_data_60['distance'])
        
        return self.vehicle_data_dict
        
    def run_simulator(self):
        self.beamng.close()
        self.beamng.open()

        time.sleep(2)
        scenario = Scenario(self.track, 'cse_891_atav_nwb', description='DDPG on North Wilkesboro')
        self.ego_vehicle = Vehicle('ego_vehicle', model=self.vehicle, license='CSE 891 ATAV', color='Red')

        scenario.add_vehicle(self.ego_vehicle, pos=self.vehicle_pos, rot_quat=self.vehicle_rot_quat)
        road_a = Road('track_rubber', rid='oval_road', default_width=self.road_width)

        with open("data/" + self.track + "_nodes.csv", mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                node = (float(row[0]), float(row[1]), float(row[2]), self.road_width)
                print(node)
                road_a.add_nodes(node)

        scenario.add_road(road_a)

        scenario.make(self.beamng)
        self.beamng.scenario.load(scenario)
        self.beamng.scenario.start()
        
        print("Press Enter when ready to proceed with the rest of the code\n")
        print("This is the time to load up Freeroam mode on the game, select your car, select the replay to train on before proceeding.\n")
        input()
        self.ego_vehicle = next(iter(self.beamng.get_current_vehicles().values()))
        self.ego_vehicle.connect(self.beamng)
        self.beamng.settings.set_deterministic(60)
        
        road_sensor = RoadsSensor('road', bng=self.beamng, vehicle=self.ego_vehicle)
        gps = GPS('gps', bng=self.beamng, vehicle=self.ego_vehicle)
        
        ultrasonic_forward = Ultrasonic('ultrasonic_forward', self.beamng, self.ego_vehicle, field_of_view_y=2, near_far_planes=(0.1, 20.1), range_direct_max_cutoff=20)

        ultrasonic_left_30 = Ultrasonic('ultrasonic_left30', self.beamng, self.ego_vehicle, dir=(0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
        ultrasonic_left_60 = Ultrasonic('ultrasonic_left60', self.beamng, self.ego_vehicle, dir=(0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)

        ultrasonic_right_30 = Ultrasonic('ultrasonic_right30', self.beamng, self.ego_vehicle, dir=(-0.30, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
        ultrasonic_right_60 = Ultrasonic('ultrasonic_right60', self.beamng, self.ego_vehicle, dir=(-0.60, -1, 0), field_of_view_y=2, near_far_planes=(0.1, 10.1), range_direct_max_cutoff=10)
        
        electrics = Electrics()
        damage = Damage()
        timer = Timer()
        self.ego_vehicle.sensors.attach('electrics', electrics)
        self.ego_vehicle.sensors.attach('damage', damage)
        
        self.vehicle_sensors = [gps, road_sensor, ultrasonic_forward, ultrasonic_left_30, ultrasonic_left_60, ultrasonic_right_30, ultrasonic_right_60]
        
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
        
        self.road_geometry = self.beamng.scenario.get_road_edges('oval_road')
        
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