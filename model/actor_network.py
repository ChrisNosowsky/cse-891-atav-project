import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Concatenate
from keras.optimizers import Adam

HIDDEN_UNITS1 = 300
HIDDEN_UNITS2 = 600

class ActorNetwork:
    def __init__(self, num_sensors, num_actions, batch_size, tau, lra):
        self.num_sensors = num_sensors
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.tau = tau
        self.lra = lra
        
        self.model, self.weights, self.state = self.create_actor_network(self.num_sensors, self.num_actions)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(self.num_sensors, self.num_actions)
        self.action_gradient = Input(dtype=tf.float32, shape=[None, self.num_actions])
        self.params_gradient = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        gradients = zip(self.params_gradient, self.weights)
        self.optimizer = Adam(self.lra)
        self.optimizer.apply_gradients(gradients)
    
    def train(self):
        pass
    
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
    
    def create_actor_network(self, num_sensors, num_actions):
        S = keras.Input(shape=[num_sensors]) 
          
        h0 = Dense(HIDDEN_UNITS1, activation='relu')(S)
        h1 = Dense(HIDDEN_UNITS2, activation='relu')(h0)
        
        Steering = Dense(1, activation='tanh')(h1)
        Acceleration = Dense(1, activation='sigmoid')(h1)
        Brake = Dense(1, activation='sigmoid')(h1)
        
        V = Concatenate()([Steering, Acceleration, Brake])        
          
        model = Model(input=S, output=V)
        return model, model.trainable_weights, S