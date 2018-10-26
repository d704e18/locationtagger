import argparse
import os
import sys
from datetime import date

import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from plotting import plot_learning_curve
from utils import load_data, load_data_and_split, load_prediction_data


def main(args):
    path = os.getcwd()
    parent = os.path.dirname(path)

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced', max_iter=100)
    if args.plot:
        x, y, x_transformer = load_data(os.path.join(parent, 'data', args.training_data))
        # Use n_jobs -1 to make use of all processors.
        plt = plot_learning_curve(
            logreg, 'Logistic regression: Accuracy / Training example', x, y.argmax(axis=1), cv=5, n_jobs=None)
        plt.show()
    else:
        train_x, train_y, validation_x, validation_y, test_x, test_y, x_transformer = load_data_and_split(
            os.path.join(parent, 'data', args.training_data))
        if args.load_model:
            logreg = joblib.load(args.load_model)
        else:
            logreg.fit(train_x, train_y.argmax(axis=1))
        print("Train score: {}".format(logreg.score(train_x, train_y.argmax(axis=1))))
        print("Validation score: {}".format(logreg.score(validation_x, validation_y.argmax(axis=1))))
        print("Test score: {}".format(logreg.score(test_x, test_y.argmax(axis=1))))

        if args.predict:
            predict_data, y, timestamps = load_prediction_data(args.training_data, args.predict, x_transformer)
            predictions = logreg.predict(predict_data)
            results = pd.DataFrame(data={'label': y, 'prediction': predictions}, index=timestamps)
            print(results)

        if args.predict_proba:
            predict_data, y, timestamps = load_prediction_data(args.training_data, 4374864809416781228, x_transformer)
            predictions = logreg.predict_proba(predict_data)
            results = pd.DataFrame(data=predictions, index=timestamps)
            print(results)

        if args.save_model:
            model_directory = os.path.join(parent, 'trained_models')
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            joblib.dump(logreg, os.path.join(model_directory, args.save_model + '.pkl'))


if __name__ == "__main__":
    a_manual_device = 4374864809416781228
    parser = argparse.ArgumentParser(description='Do some logistic regression')
    parser.add_argument('training_data', help='the training data as a csv or pkl')
    parser.add_argument('--plot', action='store_true', help='plot accuracy per examples')
    parser.add_argument('--save_model', nargs='?', const='logreg_' + date.today().strftime('%Y%m%d'), type=str,
                        help='save the model to disk')
    parser.add_argument('--load_model', help='load a pre-trained model from disk')
    parser.add_argument('--predict', nargs='?', const=a_manual_device, type=int,
                        help='predict areas for a device id')
    parser.add_argument('--predict_proba', nargs='?', const=a_manual_device, type=int,
                        help='predict the probabilities for each area for a device id')
    args = parser.parse_args(sys.argv[1:])
    main(args)
