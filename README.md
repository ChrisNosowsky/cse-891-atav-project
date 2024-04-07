# cse-891-atav-project


Project Approaches to the DNN:

1. CNN + LSTM
Capture screen data, throttle and steering inputs and then use that as the ground truth for training. Then for testing, the process will involve the same process.

2. Reinforcement learning

3. Using sensor data for training and predicting next steering/throttle input




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
