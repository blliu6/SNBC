import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from plots.plot_barriers import set_2d_labels_and_title

parameters = {'axes.labelsize': 15, 'xtick.labelsize': 10, 'ytick.labelsize': 10}
plt.rcParams.update(parameters)

plt.figure()
fig = plt.figure()
ax = fig.add_subplot(111)

zones = np.array([[-2, 2], [-1.5, 1.5]])
r = 0.4
times = 1 / (1 - r)
L = zones[:, 1] - zones[:, 0]
print(zones[:2, 0], *(zones[:2, 1] - zones[:2, 0]))
p = Rectangle(zones[:2, 0], *(zones[:2, 1] - zones[:2, 0]), linestyle='-', color=(30 / 255, 144 / 255, 255 / 255),
              linewidth=2, fill=False, label='target bound')
ax.add_patch(p)

set_2d_labels_and_title('$x_1$', '$x_2$', '')
points = (np.random.random([100, 2]) - 0.5) * L

p = Rectangle(zones[:2, 0] * times, *((zones[:2, 1] - zones[:2, 0]) * times), linestyle='--', color='g',
              linewidth=2, fill=False, label='$r_b$ times bound')
ax.add_patch(p)
points = points * times
plt.scatter(points[:, 0], points[:, 1], s=13, c='orange')

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, fontsize=12)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.savefig('img/bound_enhance_1.pdf')
# plt.savefig('img/bound_enhance_1.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.add_patch(
    Rectangle(zones[:2, 0], *(zones[:2, 1] - zones[:2, 0]), linestyle='-', color=(30 / 255, 144 / 255, 255 / 255),
              linewidth=2, fill=False, label='target bound'))
set_2d_labels_and_title('$x_1$', '$x_2$', '')
for i in range(2):
    points[:, i] = np.clip(points[:, i], *zones[i])
plt.scatter(points[:, 0], points[:, 1], s=10, c='orange')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, fontsize=12)
plt.savefig('img/bound_enhance_2.png')
# plt.savefig('img/bound_enhance_2.pdf')
plt.show()
