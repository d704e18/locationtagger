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


def run(x, y, xval, yval, hidden_units, class_weights, n_features, n_classes, epochs,
        batch_size, learning_rate):

    model = create_model(hidden_units, n_features, n_classes)
    model.compile(
        optimizer=Adam(learning_rate),
        loss="categorical_crossentropy",
        sample_weight_mode="temporal",
        metrics=['accuracy'])

    history = model.fit(
        x,
        y,
        sample_weight=class_weights,
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
        labels = np.concatenate((np.asarray(labels), np.zeros((labels.shape[0], 1))), axis=1)

        # Append the labels with dummies
        dummyLabels = np.zeros((padding, labels.shape[1]))
        dummyLabels[:, -1] = 1
        labels = np.concatenate((labels, dummyLabels), axis=0)

        x += [readings]
        y += [labels]

    return np.asarray(x), np.asarray(y)


if __name__ == "__main__":
    print("yes")

    clip = 20
    class_weights = np.ones((clip, 5))
    class_weights[-1, :] = 0
    print(class_weights)

    train, val, test, _ = utils.get_example_generator(
        "../data/trimmed-aggregated-training-data.csv",
        repeat=False,
        clip=clip)

    train_x, train_y = unpack_generator(train, clip)
    val_x, val_y = unpack_generator(val, clip)
    test_x, test_y = unpack_generator(test, clip)
    kwargs = {
        "hidden_units": 50,
        "n_features": train_x[0].shape[1],
        "n_classes": train_y[0].shape[1],
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "class_weights": class_weights
    }

    trained_model, history = run(train_x, train_y, val_x, val_y, **kwargs)
    loss, accuracy = trained_model.evaluate(test_x, test_y, verbose=0)
    print("Validation loss:     {:.2f}".format(loss))
    print("Validation accuracy: {:.2f}%".format(float(accuracy) * 100))
