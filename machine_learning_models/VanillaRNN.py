# from keras.layers import *
from keras.layers import SimpleRNN, Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import utils
import numpy as np


def create_model(hidden_units, n_features, n_classes):

    model = Sequential()
    model.add(
        SimpleRNN(
            hidden_units,
            input_shape=(None, n_features),
            return_sequences=True,
            activation='tanh',
            name="simpleRNN"))
    model.add(Dense(n_classes, name="denseOne"))
    model.add(Activation('softmax', name="softmaxOutput"))

    return model


def run(x, y, hidden_units, n_features, n_classes, epochs, batch_size,
        learning_rate):

    model = create_model(hidden_units, n_features, n_classes)
    model.compile(
        optimizer=Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=['accuracy'])

    history = model.fit(
        x, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    return model, history


if __name__ == "__main__":
    print("yes")

    x = []
    y = []
    test_x = []
    test_y = []

    clip = 20
    train, val, test, _ = utils.get_example_generator(
        "../data/trimmed-aggregated-training-data.csv",
        repeat=False,
        clip=clip)

    for coolGenerator in [train, val]:
        for reading, label in coolGenerator:
            if len(reading) < clip:
                continue
            x += [np.asarray(reading)]
            y += [np.asarray(label)]
            try:
                assert reading.shape == (20, 19)
                assert label.shape == (20, 4)
            except AssertionError:
                print(reading, reading.shape, label, label.shape)
                input("continue?")
                # x, y = None, None  # Load data here, please preprocess before

    for reading, label in test:
        if len(reading) < clip:
            continue

        test_x += [np.asarray(reading)]
        test_y += [np.asarray(label)]

    x = np.asarray(x)
    y = np.asarray(y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    kwargs = {
        "hidden_units": 500,
        "n_features": 19,
        "n_classes": 4,
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001
    }

    trained_model, history = run(x, y, **kwargs)
    loss, accuracy = trained_model.evaluate(test_x, test_y, verbose=0)
    print("Validation loss:     {:.2f}".format(loss))
    print("Validation accuracy: {:.2f}%".format(float(accuracy) * 100))
