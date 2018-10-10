import argparse
import os

from sklearn.linear_model import LogisticRegression

from machine_learning_models.plotting import plot_learning_curve
from machine_learning_models.utils import *


def main():
    path = os.getcwd()
    parent = os.path.dirname(path)

    x, y = load_data(os.path.join(parent, 'data', 'merged.pkl'))

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced', max_iter=100)

    # training
    # Use n_jobs -1 to make use of all processors.

    logreg.fit(x, y.argmax(axis=1))

    #plt = plot_learning_curve(logreg, 'Logistic regression learning curve', x, y.argmax(axis=1), cv=5, n_jobs=None)
    #plt.show()

    test = logreg.predict(x)

    print(test)
    #print("Train score: {}".format(logreg.score(x, y.argmax(axis=1))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument()
    main()
