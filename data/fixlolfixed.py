import pandas as pd
from dateutil import parser

# 2018-07-09 09:08:56
df_parser = lambda date: pd.datetime.strptime(date, '%Y-%d-%m %H:%M:%S')

data_dir = ''
data = pd.read_csv(
    data_dir + "lolfixed.csv",
    parse_dates=True,
    date_parser=df_parser,
    index_col='DateTime')


START = "START_TIMESTAMP"
END = "END_TIMESTAMP"

labels = pd.read_csv(data_dir + "simple-af1-labels.csv", index_col=0)


def df_string_to_datetime(df, columns=[START, END]):
    for column in columns:
        series = df[column]
        series = series.map(lambda x: parser.parse(x, dayfirst=False))
        df[column] = series

    return df


labels = df_string_to_datetime(labels)
labels = dict(  # allows us to access labels by ID
    [(key, df) for key, df in labels.groupby("BLUETOOTHADDRESS")])


def filter_group(df, label):
    # print(df)
    # print(df.index[0].month)
    df = df[df.index.day == label[START].iloc[0].day]
    # print(label[[START, END]])
    # input()
    return df


groups = data.groupby('DeviceID')
good_groups = []
for key, group in groups:
    if key in labels:
        label = labels[key]
        group = filter_group(group, label)
        if group.size > 0:
            good_groups += [group]
        else:
            print("Key {} was empty".format(key))

    else:
        print("Skipping key", key)

new_df = good_groups.pop()

num_groups = len(good_groups) + 1
for i, group in enumerate(good_groups):
    new_df = new_df.append(group)
    if (i + 1) % 100 == 0:
        print("Finished group {} of {}".format(i+1, num_groups))

new_df = new_df.sort_index()

new_df.to_csv("fixedlolfixed.csv")

