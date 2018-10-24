import numpy as np
import pandas as pd
import os

project_root = os.path.dirname(
    os.path.abspath(__file__)) + "/" + os.pardir + "/"
data_dir = project_root + "data/"


class FeatureAggregator:
    def __init__(self, data):
        """
        Aggregates training examples with identical timestamp and id
        data: training data such as data/training-data.csv
        """
        self.data = data

    def dfs_to_df(self, frames):
        """
        Appends a list of dataframes into a single dataframe
        assumes that the keys in each dataframe is identical
        """
        master = frames.pop()
        for frame in frames:
            master = master.append(frame)

        master = master.sort_index()
        return master

    def aggregate_person(self, person):
        """
        Aggregates readings for unique timestamps into a single observation
        """
        person_observations = person.groupby('DateTime')
        new_frames = []
        for timestamp, observation in person_observations:
            observation_dict = {}
            for key in observation:
                value = np.max(observation[key])
                observation_dict[key] = [value]

            new_frames += [
                pd.DataFrame(data=observation_dict, index=[timestamp])
            ]

        new_frames = self.dfs_to_df(new_frames)

        return new_frames

    def aggregate_features(self):
        people = self.data.groupby('Device')
        num_people = len(set(self.data['Device']))
        new_people = []

        i = 0
        for key, person in people:
            person = self.aggregate_person(person)
            new_people += [person]
            i += 1
            print("Done did person {} of {}.".format(i, num_people))

        new_data = self.dfs_to_df(new_people)
        new_data.to_csv(data_dir + "aggregated-training-data.csv")

    def default():
        """
        Returns a FeatureAggregator with reasonable defaults
        """
        data = pd.read_csv(
            data_dir + "training-set.csv",
            index_col="DateTime",
            parse_dates=True,
            dayfirst=False)

        return FeatureAggregator(data)


if __name__ == "__main__":
    fa = FeatureAggregator.default()
    fa.aggregate_features()
