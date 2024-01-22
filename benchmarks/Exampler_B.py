import numpy as np
from RL_train.Env import Example as env
from RL_train.Env import get_Env


class Example():
    def __init__(self, n, D_zones, I_zones, U_zones, f, name, f_u=0, u=0):
        if len(D_zones) != n:
            raise ValueError('The dimension of D_zones is wrong.')
        if len(I_zones) != n:
            raise ValueError('The dimension of I_zones is wrong.')
        if len(U_zones) != n:
            raise ValueError('The dimension of U_zones is wrong.')
        if len(f) != n:
            raise ValueError('The dimension of f is wrong.')
        self.n = n  # number of variables
        self.D_zones = np.array(D_zones)  # local condition
        self.I_zones = np.array(I_zones)  # initial set
        self.U_zones = np.array(U_zones)  # unsafe set
        self.f = f  # differential equation
        self.name = name  # name or identifier
        self.f_u = f_u
        self.u = u


examples = {
    1: Example(  # ex22
        n=2,
        D_zones=[[-4, 4]] * 2,
        I_zones=[[-3, -1]] + [[-3, -1]],
        U_zones=[[2, 4]] + [[1, 3]],
        f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] + u[0],
           lambda x, u: -2 * x[1] - x[0] ** 2 + u[0]],
        name='C1'),

    2: Example(  # useful ex23
        n=2,
        D_zones=[[-3, 3]] * 2,
        I_zones=[[-1, -0.9]] + [[1, 1.1]],
        U_zones=[[-2.75, -2.25]] + [[-1.75, -1.25]],
        f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u[0],
           lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7)],
        name='C2'),

    3: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[-0.1, 0]] * 2,
        U_zones=[[1.2, 1.3]] + [[-0.1, 0.1]],
        f=[
            lambda x, u: x[1],
            lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u[0],
        ],
        name='C3'),

    4: Example(  # useful ex24
        n=2,
        D_zones=[[-3, 3]] * 2,
        I_zones=[[0.5, 1.5]] * 2,
        U_zones=[[-1.6, -0.4]] * 2,
        f=[lambda x, u: x[1],
           lambda x, u: -0.5 * x[0] ** 2 - x[1] + u[0],
           ],
        name='C4'),

    5: Example(  # useful ex25
        n=2,
        D_zones=[[-2, -2]] * 2,
        I_zones=[[1, 2]] + [[-0.5, 0.5]],
        U_zones=[[-1.4, -0.6]] + [[-1.4, -0.6]],
        f=[lambda x, u: x[1] + u[0],
           lambda x, u: -x[0] + (1 / 3) * x[0] ** 3 - x[1],
           ],
        name='C5'),

    6: Example(  # useful ex26
        n=3,
        D_zones=[[-0.2, 0.2]] * 3,
        I_zones=[[-0.1, 0.1]] * 3,
        U_zones=[[-0.18, -0.15]] * 3,
        f=[lambda x, u: -x[1] + u[0],
           lambda x, u: -x[2],
           lambda x, u: -x[0] - 2 * x[1] - x[2] + x[0] ** 3,  ##--+
           ],
        name='C6'),

    7: Example(  # useful ex27
        n=3,
        D_zones=[[-4, 4]] * 3,
        I_zones=[[-1, 1]] * 3,
        U_zones=[[2, 3]] * 3,
        f=[lambda x, u: x[2] + 8 * x[1],
           lambda x, u: -x[1] + x[2],
           lambda x, u: -x[2] - x[0] ** 2 + u[0],  ##--+
           ],
        name='C7'),

    8: Example(  # useful ex28
        n=4,
        D_zones=[[-4, 4]] * 4,
        I_zones=[[-0.2, 0.2]] * 4,
        U_zones=[[-3, -1]] * 4,
        f=[lambda x, u: -x[0] - x[3] + u[0],
           lambda x, u: x[0] - x[1] + x[0] ** 2 + u[0],
           lambda x, u: -x[2] + x[3] + x[1] ** 2,
           lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
        name='C8'),

    9: Example(  # useful ex29
        n=5,
        D_zones=[[-3, 3]] * 5,
        I_zones=[[0.5, 1.5]] * 5,
        U_zones=[[-2.6, -1.4]] * 5,
        f=[
            lambda x, u: -0.1 * x[0] ** 2 - 0.4 * x[0] * x[3] - x[0] + x[1] + 3 * x[2] + 0.5 * x[3],
            lambda x, u: x[1] ** 2 - 0.5 * x[1] * x[4] + x[0] + x[2],
            lambda x, u: 0.5 * x[2] ** 2 + x[0] - x[1] + 2 * x[2] + 0.1 * x[3] - 0.5 * x[4],
            lambda x, u: x[1] + 2 * x[2] + 0.1 * x[3] - 0.2 * x[4],
            lambda x, u: x[2] - 0.1 * x[3] + u[0]
        ],
        name='C9'),

    10: Example(  # useful ex19
        n=6,
        D_zones=[[-2, 2]] * 6,
        I_zones=[[1, 2]] * 6,
        U_zones=[[-1, -0.5]] * 6,
        f=[
            lambda x, u: x[0] * x[2],
            lambda x, u: x[0] * x[4],
            lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
            lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
            lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
            lambda x, u: 2 * x[1] * x[4] + u[0]
        ],
        name='C10'),

    11: Example(  # where
        n=6,
        D_zones=[[0, 10]] * 6,
        I_zones=[[3, 3.1]] * 6,
        U_zones=[[4, 4.1]] + [[4.1, 4.2]] + [[4.2, 4.3]] + [[4.3, 4.4]] + [[4.4, 4.5]] + [[4.5, 4.6]],
        f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u[0],
           lambda x, u: -x[0] - x[1] + x[4] ** 3,
           lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
           lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
           lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
           lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
           ],
        name='C11'),

    12: Example(  # where

        n=7,
        D_zones=[[-2, 2]] * 7,
        I_zones=[[0.99, 1.01]] * 7,
        U_zones=[[1.8, 2]] * 7,
        f=[lambda x, u: -0.4 * x[0] + 5 * x[2] * x[3],
           lambda x, u: 0.4 * x[0] - x[1],
           lambda x, u: x[1] - 5 * x[2] * x[3],
           lambda x, u: 5 * x[4] * x[5] - 5 * x[2] * x[3],
           lambda x, u: -5 * x[4] * x[5] + 5 * x[2] * x[3],
           lambda x, u: 0.5 * x[6] - 5 * x[4] * x[5],
           lambda x, u: -0.5 * x[6] + u[0],
           ],
        name='C12'),

    13: Example(  # useful ex17
        n=9,
        D_zones=[[-2, 2]] * 9,
        I_zones=[[0.99, 1.01]] * 9,
        U_zones=[[1.8, 2]] * 9,
        f=[
            lambda x, u: 3 * x[2] + u[0],
            lambda x, u: x[3] - x[1] * x[5],
            lambda x, u: x[0] * x[5] - 3 * x[2],
            lambda x, u: x[1] * x[5] - x[3],
            lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
            lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
            lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
            lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
            lambda x, u: 2 * x[5] * x[7] - x[8]
        ],
        name='C13'),

    14: Example(  # useful ex18
        n=12,
        D_zones=[[-2, 2]] * 12,
        I_zones=[[-0.1, 0.1]] * 12,
        U_zones=[[0, 0.5]] * 3 + [[0.5, 1.5]] * 4 + [[-1.5, -0.5]] + [[0.5, 1.5]] * 2 + [[-1.5, -0.5]] + [[0.5, 1.5]],
        f=[
            lambda x, u: x[3],
            lambda x, u: x[4],
            lambda x, u: x[5],
            lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
            lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
            lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
            lambda x, u: x[9],
            lambda x, u: x[10],
            lambda x, u: x[11],
            lambda x, u: 9.81 * x[1],
            lambda x, u: -9.81 * x[0],
            lambda x, u: -16.3541 * x[11] + u[0]
        ],
        name='C14')
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError('The example {} was not found.'.format(name))


def get_example_by_env(e: env):
    n = e.n_obs
    D_zones = []
    for l, u in zip(e.D_zones.low, e.D_zones.up):
        D_zones.append([l, u])
    I_zones = []
    for l, u in zip(e.I_zones.low, e.I_zones.up):
        I_zones.append([l, u])
    U_zones = []
    for l, u in zip(e.U_zones.low, e.U_zones.up):
        U_zones.append([l, u])
    f = e.f[:]
    name = e.path[:e.path.find('/')]
    ex = Example(n, D_zones, I_zones, U_zones, f, name)
    return ex


if __name__ == '__main__':
    ex = get_example_by_env(get_Env(22))
    pass
