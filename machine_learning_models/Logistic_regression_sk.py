import os
import sys

from sklearn.linear_model import LogisticRegression

from machine_learning_models.plotting import plot_learning_curve
from machine_learning_models.utils import load_data, load_data_and_split


def main(args):
    path = os.getcwd()
    parent = os.path.dirname(path)

    if '--plot' in args:
        x, y = load_data(os.path.join(parent, 'data', 'merged.pkl'))
        logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced', max_iter=100)
        # Use n_jobs -1 to make use of all processors.
        plt = plot_learning_curve(logreg, 'Logistic regression learning curve', x, y.argmax(axis=1), cv=5, n_jobs=None)
        plt.show()
    else:
        logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced', max_iter=100)
        train_x, train_y, validation_x, validation_y, test_x, test_y = load_data_and_split(
            os.path.join(parent, 'data', 'merged.pkl'))
        logreg.fit(train_x, train_y.argmax(axis=1))
        print("Train score: {}".format(logreg.score(train_x, train_y.argmax(axis=1))))
        print("Validation score: {}".format(logreg.score(validation_x, validation_y.argmax(axis=1))))
        print("Test score: {}".format(logreg.score(test_x, test_y.argmax(axis=1))))


if __name__ == "__main__":
    main(sys.argv[1:])
