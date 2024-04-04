from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class CNNModel:
    def __init__(self):
        pass
    def model():
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

if __name__ == '__main__':
    cnn = CNNModel()
    cnn.model()