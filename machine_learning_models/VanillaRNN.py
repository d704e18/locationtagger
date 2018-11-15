from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam


def create_model(hidden_units, n_features, n_classes):

    model = Sequential()
    model.add(SimpleRNN(hidden_units,
                        input_shape=(None, n_features),
                        activation='tanh'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model


def run(x, y, hidden_units, n_features, n_classes, epochs, batch_size, learning_rate):

    model = create_model(hidden_units, n_features, n_classes)
    model.compile(optimizer=Adam(learning_rate),
                  loss="categorical_crossentropy",
                  metrics=['accuracy']
                  )

    history = model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1
                        )

    return model, history


if __name__ == "__main__":
    print("yes")

    x, y = None, None  # Load data here, please preprocess before

    kwargs = {
        "hidden_units": 500,
        "n_features": 19,
        "n_classes": 4,
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001
    }

    trained_model, history = run(x, y, **kwargs)
