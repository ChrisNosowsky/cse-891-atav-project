from gym import spaces
import numpy as np
import copy
import math
from collections import namedtuple
from beamng import BeamNG
from constants import *

class BeamNGEnv:

    def __init__(self, home=BEAMNG_TECH_GAME_PATH_DIR, vehicle="pickup", track="north wilkesboro", road_width=10):
        self.initial_run = True
        self.client = BeamNG(home=home, vehicle=vehicle, track=track, road_width=road_width)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
        low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
        self.observation_space = spaces.Box(low=low, high=high)
        self.stuck_counter = 0

    def step(self, u):
        this_action = self.agent_to_beamng(u)

        # Apply Action
        action_beamng = self.client.get_current_actions()

        # Steering
        action_beamng['steering'] = this_action['steering']  # in [-1, 1]
        action_beamng['throttle'] = this_action['throttle']
        action_beamng['brake'] = this_action['brake']

        # Save the previous full-obs from BeamNG for the reward calculation
        obs_pre = copy.deepcopy(self.client.vehicle_data_dict)

        # Apply the Agent's action into BeamNG
        self.client.apply_actions(action_beamng)

        # Get the current full-observation from BeamNG
        obs = self.client.poll_sensors()
        
        print("STEP OBS ", obs)
        
        # Make an obsevation from a raw observation vector from BeamNG
        self.observation = self.make_observaton(obs)

        ## REWARD SECTION ##
        sp = np.array(obs['speed_x'])
        angle = np.array(obs['angle'])
        track_pos = np.array(obs['track_pos'])
        angle_deg = math.degrees(obs['angle'])
        
        # progress = sp * (np.cos(angle_deg) - np.sin(angle_deg) - np.abs(track_pos))
        # progress = np.cos(angle_deg) - np.abs(np.sin(angle_deg)) - np.abs(obs['track_pos'])
        # progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['track_pos'])
        progress = sp*np.cos(angle_deg) - np.abs(sp*np.sin(angle_deg)) - sp * np.abs(obs['track_pos'])
        reward = progress

        print("REWARD BEFORE PENALTIES: ", reward)
        episode_terminate = False
        if obs['track_pos'] > 0 and obs['track_pos'] < 0.5:
            reward = 10000

        if obs['track_pos'] > -0.9 and obs['track_pos'] < 0.9:
            reward += 20000000

        if obs['damage'] > 6000:
            print("DAMAGE PENALTY")
            reward = -50
            episode_terminate = True
            self.client.recover_vehicle()
            
        if obs['damage'] - obs_pre['damage'] > 0:
            print("DAMAGE PENALTY")
            reward = -10

        if obs['track_pos'] >= 1 or obs['track_pos'] <= -1:
            print("OUTSIDE TRACK POS")
            # episode_terminate = True
            reward = -50
            # self.client.recover_vehicle()
        
        if round(obs_pre['track_pos'], 2) == round(obs['track_pos'], 2) and \
            obs['wheelspin'] < 0.03 and obs['gear'] != 0:
            print("DETECTING STUCK.")
            self.stuck_counter += 1
        
        if self.stuck_counter >= 10:
            print("CAR STUCK")
            self.stuck_counter = 0
            episode_terminate = True
            self.client.recover_vehicle()

        self.time_step += 1

        return self.get_obs(), reward, episode_terminate

    def reset(self):
        self.time_step = 0

        obs = self.client.poll_sensors()
        self.observation = self.make_observaton(obs)

        return self.get_obs()

    def end(self):
        self.client.close_beamng()

    def get_obs(self):
        return self.observation

    def reset_beamng(self):
        pass

    def agent_to_beamng(self, u):
        beamng_action = {'steering': u[0],
                         'throttle': u[1],
                         'brake': u[2]}
        return beamng_action

    def make_observaton(self, raw_obs):
        names = [
                'gear',
                'rpm',
                'angle',
                'trackPos',
                'speedX',
                'speedY',
                'speedZ',
                'engine_temp', 
                'wheelspin', 
                'damage',
                'track_dist_forward',
                'track_dist_right_30',
                'track_dist_right_60', 
                'track_dist_left_30',
                'track_dist_left_60'
                ]
        Observation = namedtuple('Observation', names)
        # TODO: Normalize values
        return Observation(
                    gear=np.array(raw_obs['gear'], dtype=np.float32),
                    rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                    angle=np.array(raw_obs['angle'], dtype=np.float32),
                    trackPos=np.array(raw_obs['track_pos'], dtype=np.float32),
                    speedX=np.array(raw_obs['speed_x'], dtype=np.float32),
                    speedY=np.array(raw_obs['speed_y'], dtype=np.float32),
                    speedZ=np.array(raw_obs['speed_z'], dtype=np.float32),
                    engine_temp=np.array(raw_obs['engine_temp'], dtype=np.float32),
                    wheelspin=np.array(raw_obs['wheelspin'], dtype=np.float32),
                    damage=np.array(raw_obs['damage'], dtype=np.float32),
                    track_dist_forward=np.array(raw_obs['track_dist_forward'], dtype=np.float32),
                    track_dist_right_30=np.array(raw_obs['track_dist_right_30'], dtype=np.float32),
                    track_dist_right_60=np.array(raw_obs['track_dist_right_60'], dtype=np.float32),
                    track_dist_left_30=np.array(raw_obs['track_dist_left_30'], dtype=np.float32),
                    track_dist_left_60=np.array(raw_obs['track_dist_left_60'], dtype=np.float32)
                    )