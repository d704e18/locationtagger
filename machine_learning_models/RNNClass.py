from keras.layers import SimpleRNN, Dense, Activation, Masking, BatchNormalization
import keras.layers as kl
from keras.models import Sequential
import keras
import numpy as np
import os
from Visualization import save_hist_plot


class RNNClass:

    def __init__(self, save_path, input_shape=(20, 19), classes=4):

        # Fixed model parameters
        self.classes = classes
        self.input_shape = input_shape
        self.loss = keras.losses.categorical_crossentropy
        self.optimizer = keras.optimizers.Adam
        self.epochs = 500
        self.callbacks = []


        # Data sets
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None

        self.save_path = save_path

        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def set_dataset(self, data_tuple):
        self.X_train, self.Y_train, self.X_val, self.Y_val = data_tuple

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def run(self, name, hidden_units, n_layers, dropout, learning_rate, batch_size=256, epochs=500):

        model = self._build_model(hidden_units, n_layers, dropout)
        history = self._train(model, learning_rate, batch_size, epochs)

        save_hist_plot(history, name, self.save_path)

        best_model = keras.models.load_model(self.save_path + "/{}".format(name))

        return best_model.evaluate(self.X_val, self.Y_val, batch_size=512)[1] * 100


    def _train(self, model, learning_rate, batch_size, epochs):
        # Compile model
        model.compile(
            optimizer=self.optimizer(learning_rate),
            loss=self.loss,
            metrics=['accuracy'])

        history = model.fit(
            self.X_train,
            self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.Y_val),
            shuffle=True,
            callbacks=self.callbacks
        )

        return history

    def _build_model(self, hidden_units, n_layers, dropout):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=self.input_shape))

        for i in range(0, n_layers):

            units = int(hidden_units*(1-(i*0.2)))
            units = units if units > 0 else 50

            dp = dropout if i < 3 else 0

            model.add(
                SimpleRNN(units,
                          return_sequences=True,
                          dropout=dp,
                          activation="tanh",
                          name="RNN_{}".format(i)))

        model.add(Dense(self.classes, name="denseOutput"))
        model.add(Activation('softmax', name="softmaxOutput"))

        return model
