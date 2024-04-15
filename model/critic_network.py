import tensorflow as tf
import keras
import os
from keras.models import Model
from keras.layers import Dense, Concatenate
from keras.optimizers.legacy import Adam
from tensorflow.python.keras import backend as K
import tensorflow.python.util.deprecation as deprecation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')
HIDDEN_UNITS1 = 300
HIDDEN_UNITS2 = 600

class CriticNetwork:
    def __init__(self, sess: tf.compat.v1.Session, num_sensors, num_actions, batch_size, tau, lrc):
        self.sess = sess
        self.num_sensors = num_sensors
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.tau = tau
        self.lrc = lrc
        
        K.set_session(sess)
        
        self.model, self.action, self.state = self.create_critic_network(self.num_sensors, self.num_actions)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(self.num_sensors, self.num_actions)
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.compat.v1.initialize_all_variables())
    
    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]
    
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
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(learning_rate=self.lrc)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S