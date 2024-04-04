from keras.models import Sequential
from keras.layers import Dense

class DNNModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def model(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_shape=(self.x_train.shape[1],)))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()

if __name__ == '__main__':
    dnn = DNNModel()
    dnn.model()
    