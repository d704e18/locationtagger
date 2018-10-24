import numpy as np
import pandas as pd
import os

project_root = os.path.dirname(
    os.path.abspath(__file__)) + "/" + os.pardir + "/"
data_dir = project_root + "data/"


class Grouper:
    def __init__(self, df, seconds):
        self.df = df
        self.seconds = seconds

    def group(self):
        return None


if __name__ == "__main__":
    None

