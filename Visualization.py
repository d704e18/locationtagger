import matplotlib.pyplot as plt
from utils import cd
import os


def save_hist_plot(history, name="test", path=None):

    train_errors = history.history['loss']
    val_errors = history.history['val_loss']

    plt.style.use('bmh')
    plt.plot(range(len(train_errors)), train_errors, 'g-', label="Train")
    plt.plot(range(len(val_errors)), val_errors, 'r-', label="Val")
    plt.legend()

    if path is None:
        path = os.getcwd()+"/Data"

    with cd(path):
        plt.savefig("Train_val_graph_{}".format(name))
        plt.clf()
