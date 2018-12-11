import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

filename = "without-staff-secondsout"
seconds = np.loadtxt(filename)
print(seconds.shape)
print(np.mean(seconds))
print(np.median(seconds))
print(np.unique(seconds, return_counts=True))
# sns.distplot(seconds, bins=50)
# plt.show()
v = np.unique(seconds, return_counts=True)

x = [0]
y = [0]

for v1, v2 in zip(*v):
    x += [v1]
    y += [v2]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# line, = ax.plot(x, y)
y = np.cumsum(y)
x = np.arange(len(y)) + 1
for i, v in enumerate(y):
    print("{}: {}/{}, {}%".format(i, v, y[-1], (v / y[-1]) * 100))
a = [pow(10, i) for i in range(10)]
line, = ax.plot(x, y)
# ax.set_yscale('log')
# plt.xlim(0, 500)
# plt.plot(x, y)
plt.show()
