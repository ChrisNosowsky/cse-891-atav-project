import gym
from gym import spaces
import numpy as np
import copy
import os
import time
from collections import namedtuple
from beamng import BeamNG

class BeamNGEnv:
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self):
        self.initial_run = True
        self.client = BeamNG()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
        low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
        self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        client = self.client

        this_action = self.agent_to_beamng(u)

        # Apply Action
        action_beamng = client.R.d

        # Steering
        action_beamng['steering'] = this_action['steering']  # in [-1, 1]
        action_beamng['throttle'] = this_action['throttle']
        action_beamng['brake'] = this_action['brake']

        
        # Save the privious full-obs from BeamNG for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into BeamNG
        client.respond_to_server()
        # Get the response of BeamNG
        client.get_servers_input()

        # Get the current full-observation from BeamNG
        obs = client.S.d

        # Make an obsevation from a raw observation vector from BeamNG
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        episode_terminate = False

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self):
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

        obs = self.client.get_observations()
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        self.client.close_beamng()

    def get_obs(self):
        return self.observation

    def reset_beamng(self):
        pass

    def agent_to_beamng(self, u):
        beamng_action = {'steering': u[0]}

        if self.throttle is True:  # throttle action is enabled
            beamng_action.update({'throttle': u[1]})
            beamng_action.update({'brake': u[2]})

        return beamng_action

    def make_observaton(self, raw_obs):
        names = ['focus',
                    'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                    'opponents',
                    'rpm',
                    'track', 
                    'trackPos',
                    'wheelSpinVel']
        Observation = namedtuple('Observation', names)
        return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                            speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                            speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                            speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                            angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                            damage=np.array(raw_obs['damage'], dtype=np.float32),
                            opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                            rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                            track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                            trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                            wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))