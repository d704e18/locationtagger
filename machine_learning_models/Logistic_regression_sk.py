import argparse
import os
import sys
from datetime import date

import numpy
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier

from plotting import plot_learning_curve
from utils import load_data, load_data_and_split, load_prediction_data, one_hot


def main(args):
    path = os.getcwd()
    parent = os.path.dirname(path)

    logreg = SGDClassifier(loss='log', shuffle=True, max_iter=100, penalty=None, class_weight='balanced')
    if args.plot:
        x, y, x_transformer = load_data(os.path.join(parent, 'data', args.training_data))
        # Use n_jobs=-1 to make use of all processors.
        plt = plot_learning_curve(
            logreg, 'Logistic regression: Accuracy / Training example', x, y.argmax(axis=1), cv=5, n_jobs=-1)
        plt.show()
    else:
        train_x, train_y_non_one_hot, validation_x, validation_y, test_x, test_y, x_transformer = load_data_and_split(
            os.path.join(parent, 'data', args.training_data))
        train_y = one_hot(train_y_non_one_hot)
        if args.load_model:
            logreg = joblib.load(args.load_model)
        else:
            logreg.fit(train_x, train_y.argmax(axis=1))
        print('Train score: {}'.format(logreg.score(train_x, train_y.argmax(axis=1))))
        print('Validation score: {}'.format(logreg.score(validation_x, validation_y.argmax(axis=1))))
        print('Test score: {}'.format(logreg.score(test_x, test_y.argmax(axis=1))))

        if args.predict or args.predict_proba:
            predict_data, y, timestamps = load_prediction_data(args.training_data, args.predict, x_transformer)

            if args.predict:
                predictions = logreg.predict(predict_data)
                results = pd.DataFrame(data={'label': y, 'prediction': predictions}, index=timestamps)
                print(results)

            if args.predict_proba:
                predictions = logreg.predict_proba(predict_data)
                results = pd.DataFrame(data=predictions, index=timestamps)
                print(results)

        if args.confidence:
            probabilities = logreg.predict_proba(train_x)
            predictions = logreg.predict(train_x)
            probas_predictions_labels = numpy.concatenate(
                (probabilities, predictions.reshape(-1, 1), train_y_non_one_hot), axis=1)
            df = pd.DataFrame(probas_predictions_labels, columns=['a0', 'a1', 'a2', 'a3', 'prediction', 'label'])
            correct_predictions = df.loc[df['prediction'] == df['label']]
            highest_probas_correct = correct_predictions[['a0', 'a1', 'a2', 'a3']].max(axis=1)
            highest_probas_correct_avg = numpy.average(highest_probas_correct)
            highest_probas_correct_std = numpy.std(highest_probas_correct)

            highest_probas = numpy.max(probabilities, axis=1)
            highest_probas_avg = numpy.average(highest_probas)
            highest_probas_std = numpy.std(highest_probas)

            print('Highest avg. probability:', highest_probas_avg)
            print('Highest probability std:', highest_probas_std)
            print('Highest correct probabilities avg:', highest_probas_correct_avg)
            print('Highest correct probabilities std:', highest_probas_correct_std)

        if args.save_model:
            model_directory = os.path.join(parent, 'trained_models')
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            joblib.dump(logreg, os.path.join(model_directory, args.save_model + '.joblib'))


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
    parser.add_argument('--confidence', action='store_true', help='calculate confidence')
    args = parser.parse_args(sys.argv[1:])
    main(args)
