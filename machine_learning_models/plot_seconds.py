import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

filename = "with-staff-seconds.out"
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
# ax.set_title("cumulative observation interval")
ax.set_ylabel("#cumulative observations")
ax.set_xlabel("#seconds")
# line, = ax.plot(x, y)
y = np.cumsum(y)
x = np.arange(len(y)) + 1
# for i, v in enumerate(y):
#     print("{}: {}/{}, {}%".format(i, v, y[-1], (v / y[-1]) * 100))
a = [pow(10, i) for i in range(10)]
line, = ax.plot(x, y)
# ax.set_yscale('log')
plt.xlim(0, 400)
# plt.plot(x, y)
plt.savefig("with-staff.png", bbox_inches='tight')
