import numpy as np
import json
import tensorflow as tf
from model.actor_network import ActorNetwork
from model.critic_network import CriticNetwork
from model.replay_buffer import ReplayBuffer
from model.ou import OU
from gym_beamng import BeamNGEnv

NUM_SENSORS = 15
NUM_ACTIONS = 3
ACTOR_MODEL = 'actor_model.h5'
ACTOR_MODEL_JSON = 'actor_model.json'
CRITIC_MODEL = 'critic_model.h5'
CRITIC_MODEL_JSON = 'critic_model.json'

np.random.seed(1337)

class DDPGModel:
    def __init__(self, num_sensors, num_actions=3, train_rl=0, buffer_size=100000, batch_size=32, gamma=0.99, tau=0.001, lra=0.0001, lrc=0.001):
        self.num_sensors = num_sensors
        self.num_actions = num_actions
        self.train_rl = train_rl
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lra = lra
        self.lrc = lrc
        self.reward = 0
        self.explore = 100000.
        self.episode_count = 2000
        self.max_steps = 100000
        self.epsilon = 1

    # def set_gpus():
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     if gpus:
    #         try:
    #             for gpu in gpus:
    #                 tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #         except RuntimeError as e:
    #             print(e)

    def get_actor(self):
        return ActorNetwork(self.num_sensors, self.num_actions, self.batch_size, self.tau, self.lra)
        
    def get_critic(self):
        return CriticNetwork(self.num_sensors, self.num_actions, self.batch_size, self.tau, self.lrc)
    
    def get_replay_buffer(self):
        return ReplayBuffer(self.buffer_size)
    
    def model(self):
        total_reward = 0
        step = 0
        done = False
        
        actor = self.get_actor()
        critic = self.get_critic()
        buff = self.get_replay_buffer()
        
        env = BeamNGEnv()
        env.client.run_simulator()
        
        try:
            actor.model.load_weights(ACTOR_MODEL)
            critic.model.load_weights(CRITIC_MODEL)
            actor.target_model.load_weights(ACTOR_MODEL)
            critic.target_model.load_weights(CRITIC_MODEL)
            print("Weight load successfully")
        except:
            print("Cannot find the weight")
        
        
        for i in range(self.episode_count):
            print("Episode : " + str(i) + "/" + str(self.episode_count))
            
            obs = env.reset()   # Initial observation
            
            sensors_t = np.hstack((obs.gear, obs.rpm, obs.angle, obs.trackPos, obs.speedX, obs.speedY, obs.speedZ, obs.engine_temp, 
                                   obs.wheelspin, obs.damage, obs.track_dist_forward, 
                                   obs.track_dist_right_30, obs.track_dist_right_60, obs.track_dist_left_30, 
                                   obs.track_dist_left_60))
            for j in range(self.max_steps):
                loss = 0.0
                self.epsilon -= 1.0 / self.explore
                actions = np.zeros([1, self.num_actions])
                noises = np.zeros([1, self.num_actions])
                original_actions = actor.model.predict(sensors_t.reshape(1, -1))
                
                OU_Steering = OU(original_actions[0][0], 0.0 , 0.60, 0.3)
                OU_Throttle = OU(original_actions[0][1], 0.5 , 1.00, 0.1)
                OU_Brake = OU(original_actions[0][2], -0.1 , 1.00, 0.05)
                
                noises[0][0] = self.train_rl * max(self.epsilon, 0) * OU_Steering.call_func()   # STEERING
                noises[0][1] = self.train_rl * max(self.epsilon, 0) * OU_Throttle.call_func()   # THROTTLE
                noises[0][2] = self.train_rl * max(self.epsilon, 0) * OU_Brake.call_func()      # BRAKE


                actions[0][0] = original_actions[0][0] + noises[0][0]
                actions[0][1] = original_actions[0][1] + noises[0][1]
                actions[0][2] = original_actions[0][2] + noises[0][2]
            
                obs, reward_t, done = env.step(actions[0])
                
                sensors_t1 = np.hstack((obs.gear, obs.rpm, obs.angle, obs.trackPos, obs.speedX, obs.speedY, obs.speedZ, obs.engine_temp, 
                                    obs.wheelspin, obs.damage, obs.track_dist_forward, 
                                    obs.track_dist_right_30, obs.track_dist_right_60, obs.track_dist_left_30, 
                                    obs.track_dist_left_60))
                buff.add(sensors_t, actions[0], reward_t, sensors_t1, done)
                
                batch = buff.get_batch(self.batch_size)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])
                
                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
                
                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + self.gamma*target_q_values[k]
                
                if (self.train_rl):
                    loss += critic.model.train_on_batch([states, actions], y_t)
                    a_for_grad = actor.model.predict(states)
                    grads = critic.gradients(states, a_for_grad)
                    actor.train(states, grads)
                    actor.target_train()
                    critic.target_train()

                total_reward += reward_t
                sensors_t = sensors_t1
                
                print("Episode", i, "Step", step, "Action", actions, "Reward", reward_t, "Loss", loss)
                
                step += 1
                if done:
                    print("DONE. TERMINATE.")
                    break
            
            if np.mod(i, 3) == 0:
                if (self.train_rl):
                    print("Saving model")
                    actor.model.save_weights(ACTOR_MODEL, overwrite=True)
                    with open(ACTOR_MODEL_JSON, 'w') as outputfile:
                        json.dump(actor.model.to_json(), outputfile)
                        
                    critic.model.save_weights(CRITIC_MODEL, overwrite=True)
                    with open(CRITIC_MODEL_JSON, 'w') as outputfile:
                        json.dump(critic.model.to_json(), outputfile)
            
            print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
            print("Total Step: " + str(step))
            print("")
            
        print("Finished RL")
                
                
if __name__ == '__main__':
    ddpg = DDPGModel(NUM_SENSORS, NUM_ACTIONS)
    ddpg.model()
    