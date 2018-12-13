import keras
import os
from machine_learning_models import utils

if __name__ == "__main__":
    seq_length = 20

    name = "sec_real"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    results_path = dir_path + "/results/" + name
    data_path = dir_path + "/data/trimmed-aggregated-training-data.csv"

    print("Loading and preprocessing data")

    dict_data = {
        "time_differences": [0, 30, 60, 300, 600, 1800, 3600],
        "time_resolution": 1
    }

    train, val, test, _ = utils.get_rnn_data(data_path, clip=seq_length, pad=True)
    train_x, train_y = train
    val_x, val_y = val
    test_x, test_y = test

    for i in range(1, 63):
        model_name = "Vanilla_RNN_sec_real_param_{}".format(i)
        model_path = "{}/{}".format(results_path, model_name)

        print("Loading model {}".format(model_name))
        model = keras.models.load_model(model_path)

        print("Evaluating on training set")
        train_eval = model.evaluate(train_x, train_y, batch_size=512)

        print("Evaluating on validation set")
        val_eval = model.evaluate(val_x, val_y, batch_size=512)

        print("Evaluating on testing set")
        test_eval = model.evaluate(test_x, test_y, batch_size=512)

        print("Printing results to csv")
        with utils.cd(results_path):
            with open("{}_evaluations.csv".format(name), "a") as the_file:

                the_file.write("\n{}: \n Train set accuracy: {} \n Train set loss: {} \n"
                               .format(model_name, train_eval[1], train_eval[0]))

                the_file.write(" Validation accuracy: {} \n Validation loss: {} \n"
                               .format(val_eval[1], val_eval[0]))

                the_file.write(" Test accuracy: {} \n Validation loss: {} \n"
                               .format(test_eval[1], test_eval[0]))
