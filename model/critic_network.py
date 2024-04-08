import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Concatenate
from keras.optimizers import Adam

HIDDEN_UNITS1 = 300
HIDDEN_UNITS2 = 600

class CriticNetwork:
    def __init__(self, num_sensors, num_actions, batch_size, tau, lrc):
        self.num_sensors = num_sensors
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.tau = tau
        self.lrc = lrc
        self.model, self.action, self.state = self.create_critic_network(self.num_sensors, self.num_actions)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(self.num_sensors, self.num_actions)
        self.action_grads = tf.gradients(self.model.output, self.action)
    
    def gradients(self, states, actions):
        pass
    
    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
    
    def create_critic_network(self, num_sensors, num_actions):
        S = keras.Input(shape=[num_sensors])  
        A = keras.Input(shape=[num_actions],name='action2')   
        w1 = Dense(HIDDEN_UNITS1, activation='relu')(S)
        a1 = Dense(HIDDEN_UNITS2, activation='linear')(A) 
        h1 = Dense(HIDDEN_UNITS2, activation='linear')(w1)
        h2 = Concatenate()([h1,a1])
        h3 = Dense(HIDDEN_UNITS2, activation='relu')(h2)
        V = Dense(num_actions,activation='linear')(h3)   
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.lrc)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S