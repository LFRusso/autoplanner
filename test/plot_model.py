from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Conv2D(258, (3,3), input_shape=(11, 11, 6)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(.2))

model.add(Conv2D(258, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(4, activation="linear"))
model.compile(loss="mse", optimizer=Adam(learning_rate=.001), metrics=["accuracy"])

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, rankdir="TB",)