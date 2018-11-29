from src import SingleParam, Tuner, ParamConfig, ParamLog
from machine_learning_models.RNNClass import RNNClass
import machine_learning_models.utils as utils
import os
import numpy as np


def stopper(trials):
    if trials > 200:
        return True

    return False


if __name__ == "__main__":
    seq_length = 20

    name = "test"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    save_path = dir_path + "/results/" + name
    data_path = dir_path + "/data/trimmed-aggregated-training-data.csv"

    print("Loading and preprocessing data")
    train, val, test, _ = utils.get_rnn_data(data_path, clip=seq_length, pad=True)
    train_x, train_y = train
    val_x, val_y = val
    test_x, test_y = test
    data_tuple = (train_x, train_y, val_x, val_y)

    features = train_x[0].shape[1]

    # INIT PARAMETERS
    print("Initializing parameter configuration")

    params_RNN = (
        SingleParam("learning_rate", output_type="double", value_range=(0.0005, 0.001), scaling="log"),
        SingleParam("n_layers", output_type="discrete", value_range=[1, 2, 3]),
        SingleParam("hidden_units", output_type="discrete", value_range=[50, 100, 200, 300]),
        SingleParam("dropout", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.1)
    )

    p_config = ParamConfig()
    rescaler_functions_RNN, _ = p_config.make_rescale_dict(params_RNN)

    print("Initializing model creator and hyperparameter tuner")

    # INIT RNN MODEL
    SameShit = RNNClass(save_path=save_path, input_shape=(seq_length, features))

    # INIT HYPER PARAMETER TUNER
    DifferentHyperParameter = Tuner("Vanilla_RNN_{}".format(name), sam=SameShit, param_config=params_RNN,
                                    suggestors="RandomSearch", save_path=save_path)


    # TUNING
    print("Tuning/training")
    SameShit.set_dataset(data_tuple)
    DifferentHyperParameter.tune(stopper)
