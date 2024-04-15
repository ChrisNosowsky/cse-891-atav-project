import keras
import os
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Concatenate
from tensorflow.python.keras import backend as K
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

HIDDEN_UNITS1 = 300
HIDDEN_UNITS2 = 600
class ActorNetwork:
    def __init__(self, sess: tf.compat.v1.Session, num_sensors, num_actions, batch_size, tau, lra):
        self.sess = sess
        self.num_sensors = num_sensors
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.tau = tau
        self.lra = lra
        
        K.set_session(sess)
        
        self.model, self.weights, self.state = self.create_actor_network(self.num_sensors, self.num_actions)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(self.num_sensors, self.num_actions)
        # self.action_gradient = Input(dtype=tf.float32, shape=[None, self.num_actions])
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, self.num_actions])
        # reshaped_output = tf.reshape(self.model.output, (-1, 1))
        self.params_gradient = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        gradients = zip(self.params_gradient, self.weights)
        # self.optimizer = Adam(self.lra)
        # self.optimizer.apply_gradients(gradients)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lra).apply_gradients(gradients)
        self.sess.run(tf.compat.v1.initialize_all_variables())
    
    def train(self, states, action_grads):
        self.sess.run(self.optimizer, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })
    
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
          
        model = Model(S, V)
        return model, model.trainable_weights, S