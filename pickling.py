import pandas
import os

# TODO: change paths
BASEPATH = os.path.dirname(os.path.abspath(__file__)) + '/'
AF1_BASEPATH = BASEPATH + "data/af1/"
PICKLE_BASEPATH = BASEPATH + "data/pickle/"
AF1_ZONES_PATH = BASEPATH + "sensors/sensors_af1.pkl"

DAY1 = "20180906"
DAY2 = "20180907"
DAY3 = "20180908"
DAY4 = "20180909"
DAY5 = "20180910"
DAY6 = "20180911"
DAY7 = "20180912"

af1_zones = pandas.read_pickle(AF1_ZONES_PATH)


def filter_af1_data(df, filename):
    if not os.path.exists(AF1_BASEPATH):
        os.mkdir(AF1_BASEPATH)
    # pandas.DataFrame.to_csv(
    #     df[df.SensorID.isin(af1_zones.ID)],
    #     AF1_BASEPATH + filename + ".csv"
    # )
    df[df.SensorID.isin(af1_zones.ID)].to_csv(AF1_BASEPATH + filename + ".csv")


data_from_day1 = pandas.read_pickle(PICKLE_BASEPATH + DAY1 + ".pkl")
filter_af1_data(data_from_day1, DAY1)
print("Done filtering day1")
data_from_day2 = pandas.read_pickle(PICKLE_BASEPATH + DAY2 + ".pkl")
filter_af1_data(data_from_day2, DAY2)
print("Done filtering day2")
data_from_day3 = pandas.read_pickle(PICKLE_BASEPATH + DAY3 + ".pkl")
filter_af1_data(data_from_day3, DAY3)
print("Done filtering day3")
data_from_day4 = pandas.read_pickle(PICKLE_BASEPATH + DAY4 + ".pkl")
filter_af1_data(data_from_day4, DAY4)
print("Done filtering day4")
data_from_day5 = pandas.read_pickle(PICKLE_BASEPATH + DAY5 + ".pkl")
filter_af1_data(data_from_day5, DAY5)
print("Done filtering day5")
data_from_day6 = pandas.read_pickle(PICKLE_BASEPATH + DAY6 + ".pkl")
filter_af1_data(data_from_day6, DAY6)
print("Done filtering day6")
data_from_day7 = pandas.read_pickle(PICKLE_BASEPATH + DAY7 + ".pkl")
filter_af1_data(data_from_day7, DAY7)
print("Done filtering day7")

