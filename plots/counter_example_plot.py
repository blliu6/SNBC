import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from benchmarks.Exampler_B import get_example_by_name
from plots.plot_barriers import barrier_3d, set_3d_labels_and_title
from plots.plot_fcns import plot_square_sets


def gen_sample(center, sample_nums):
    result = []
    eps = 0.6
    for i in range(sample_nums):
        rd = (np.random.random(2) - 0.5) * eps
        rd = rd + center
        result.append(rd)
    return np.array(result)


ex = get_example_by_name('F1')
x = sp.symbols(['x1', 'x2'])
B = "-0.105024223731795*x1**2 - 0.174331989689821*x1*x2 - 0.2335244231776*x1 + 0.0287380828434669*x2**2 - " \
    "0.657170092537081*x2 + 0.780483266656108"
B = sp.sympify(B)
eps = 0.6
X = np.linspace(ex.I_zones[0, 0] - eps, ex.I_zones[0, 1] + eps, 100)
Y = np.linspace(ex.I_zones[1, 0] - eps, ex.I_zones[1, 1] - eps / 2, 100)
x0, x1 = np.meshgrid(X, Y)
lambda_b = sp.lambdify(x, str(B), modules=['numpy'])
plot_b = -lambda_b(x0, x1)

ax = barrier_3d(x0, x1, plot_b, True)
# Init and Unsafe sets
ax = plot_square_sets(ax, ex.I_zones[0], ex.I_zones[1], 'g', 'Initial Set')
set_3d_labels_and_title(ax, '$x_1$', '$x_2$', 'B', ' Candidate B')
center = [0, 1]
ax.scatter(center[:1], center[1:], [0], c='r', s=100)  # marker='^'
res = gen_sample(center, 20)

ax.scatter(res[:, 0], res[:, 1], [0] * res.shape[0], c='b', s=20)
plt.savefig('img/counter_example.pdf')
plt.show()
