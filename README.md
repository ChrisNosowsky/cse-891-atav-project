# cse-891-atav-project



## Overview
Project Approach: Reinforcement Learning with DDPG

This project focuses on developing an autonomous racecar agent capable of navigating various racetracks within the BeamNG.drive simulation environment. Utilizing Deep Deterministic Policy Gradient (DDPG), an algorithm in reinforcement learning, the project aims to train a highly efficient and competitive racecar that can autonomously handle the complexities of high-speed racing.

## Learning Approach
- Deep Deterministic Policy Gradient (DDPG): DDPG is chosen for its effectiveness in handling high-dimensional, continuous action spaces typical in driving applications. It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network) to learn policies that dictate the continuous steering, throttling, and braking actions necessary for racecar control.
- Agent Architecture: The agent consists of two main components:
    - Actor: Determines the best action to take given the current state of the car.
    - Critic: Evaluates the action output by the actor based on the reward structure defined for racing. The critic helps in fine-tuning the policy by providing necessary adjustments.


## Technologies
- BeamNG.drive: A highly realistic vehicle simulation that provides the platform for training and evaluating our autonomous agent.
- Python: The primary programming language used for creating the DDPG agent.
- TensorFlow/Keras: Utilized for building and training the neural network models that underpin both the actor and critic components of the DDPG algorithm.


## How to Run
### Prerequisites
You will need the following to get this to run:
- BeamNG.Tech license (For the various sensors)
- Python 3.9.10
- Tensorflow + Keras 2.15.0
- Beamngpy 1.28
- North Wilkeboro Mod (https://www.beamng.com/threads/paws-nascar-tracks.87837/)

Other requirements are defined in the `requirements.txt`

Run the following command from the root:

```python
python ddpg.py
```

By default, this will open up North Wilkeboro track and start the scenario. You will have to press `ENTER` twice in the terminal to advance through the loading of the sensors + various road input data.

If you want to customize the track and vehicle that get loaded in you can modify the following in the **main** function of the `ddpg.py` file:

```python
if __name__ == '__main__':
    #### CONFIGURE HERE! ####
    vehicle = "moonhawk"        # MODIFY HERE!
    track = "north wilkesboro"  # MODIFY HERE!
    #########################
    
    if os.path.exists(BEAMNG_TECH_GAME_PATH_DIR):
        ddpg = DDPGModel(NUM_SENSORS, NUM_ACTIONS, train_rl=1, 
                         home=BEAMNG_TECH_GAME_PATH_DIR, vehicle=vehicle, track=track)
    else:
        ddpg = DDPGModel(NUM_SENSORS, NUM_ACTIONS, train_rl=1, home=MISHA_BEAMNG_TECH_GAME_PATH_DIR, 
                         vehicle=vehicle, track=track)
    ddpg.model()
```

## Notes
Best AI to train per track

Charlotte Roval: RC9


## Thought process
1. model_ddpg calls gym_beamng.py BeamNGEnv class to start up BeamNG environment
2. model_ddpg next calls run_simulator from beamng.py to setup game state and prepare for training
3. once game state is "ready", model_ddpg runs through episodes and gets first observation from game (poll sensor data)
4. next, actor model looks at game state and predicts and action to do (steer left, right, etc.)
5. critic model critizes the actor
6. Next state is made
7. Process repeats

`beamng_standalone.py` = For testing purposes only
