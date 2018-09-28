import os
from machine_learning_models.utils import *
from sklearn.linear_model import LogisticRegression
import numpy as np

def main():

    path = os.getcwd()
    parent = '/'.join(path.split('/')[:-1])

    train_x, train_y, validation_x, validation_y, test_x, test_y = load_data(parent+"/data/whole_week.pkl")

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced',
                                max_iter=100)

    # training
    logreg.fit(train_x, train_y.argmax(axis=1))

    print("Validation score: {}".format(logreg.score(validation_x, validation_y.argmax(axis=1))))
    print("Test score: {}".format(logreg.score(test_x, test_y.argmax(axis=1))))

if __name__ == "__main__":
    main()