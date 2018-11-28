# from keras.layers import *
from keras.layers import SimpleRNN, Dense, Activation, Flatten, Embedding, Masking, GRU
from keras.models import Sequential
from keras.optimizers import Adam
import keras
import utils
import numpy as np


def create_model(hidden_units, n_features, n_classes):

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(20, n_features)))
    model.add(
        SimpleRNN(hidden_units,
                  return_sequences=True,
                  dropout=0.2,
                  activation='tanh',
                  name="RNN1"))
    model.add(Dense(n_classes, name="denseOne"))
    model.add(Activation('softmax', name="softmaxOutput"))

    return model


def run(x, y, xval, yval, hidden_units, n_features, n_classes, epochs,
        batch_size, learning_rate):

    model = create_model(hidden_units, n_features, n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=['accuracy'])

    history = model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(xval, yval))

    return model, history


def unpack_generator(gen, clip):
    x = []
    y = []
    for readings, labels in gen:
        # Calculate the padding amount
        padding = clip - len(readings)
        # Generate the 0-values for the training data
        zeros = np.zeros((padding, readings.shape[1]))
        readings = np.concatenate((np.asarray(readings), zeros), axis=0)

        # Add a column of 0's as dummy labels
        #labels = np.concatenate((np.asarray(labels), np.zeros((labels.shape[0], 1))), axis=1)

        # Append the labels with dummies
        dummyLabels = np.zeros((padding, labels.shape[1]))
        #dummyLabels[:, -1] = 1
        labels = np.concatenate((labels, dummyLabels), axis=0)

        x += [readings]
        y += [labels]

    return np.asarray(x), np.asarray(y)


if __name__ == "__main__":
    print("yes")

    clip = 20

    train, val, test, _ = utils.get_example_generator(
        "../data/trimmed-aggregated-training-data.csv",
        repeat=False,
        clip=clip)

    train_x, train_y = unpack_generator(train, clip)
    val_x, val_y = unpack_generator(val, clip)
    test_x, test_y = unpack_generator(test, clip)
    kwargs = {
        "hidden_units": 100,
        "n_features": train_x[0].shape[1],
        "n_classes": train_y[0].shape[1],
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 0.001
    }

    print(kwargs)

    trained_model, history = run(train_x, train_y, val_x, val_y, **kwargs)
    loss, accuracy = trained_model.evaluate(test_x, test_y, verbose=0)
    print("Test loss:     {:.2f}".format(loss))
    print("Test accuracy: {:.2f}%".format(float(accuracy) * 100))
