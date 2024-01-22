import numpy as np
import torch
import time
import torch
import os


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = sigmoid(self.layer2(h_1))
        return out


class Zones:
    def __init__(self, shape, center=None, r=0.0, low=None, up=None, inner=True):
        self.shape = shape
        self.inner = inner
        if shape == 'ball':
            self.center = np.array(center)
            self.r = r
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2
            self.r = sum(((self.up - self.low) / 2) ** 2)


class Example:
    def __init__(self, n_obs, u_dim, D_zones, I_zones, G_zones, U_zones, f, u, path, dense, units, activation, id, k):
        self.n_obs = n_obs
        self.u_dim = u_dim
        self.D_zones = D_zones
        self.I_zones = I_zones
        self.G_zones = G_zones
        self.U_zones = U_zones
        self.f = f
        self.u = u
        self.path = path
        self.dense = dense
        self.units = units
        self.activation = activation
        self.k = k
        self.id = id


class Env:
    def __init__(self, example):
        self.n_obs = example.n_obs
        self.u_dim = example.u_dim
        self.D_zones = example.D_zones
        self.I_zones = example.I_zones
        self.G_zones = example.G_zones
        self.U_zones = example.U_zones
        self.f = example.f
        self.path = example.path
        self.u = example.u

        self.dense = example.dense
        self.units = example.units
        self.activation = example.activation
        self.id = example.id
        self.dt = 0.001
        self.k = example.k

        self.dic = dict()
        # self.path = self.path.split('/')[0] + '_with_lidao' + '/' + self.path.split('/')[1]
        print(self.path)

    def reset(self, s=None):

        if s is not None:
            self.s = np.array(s)
        else:
            self.s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
            if self.I_zones.shape == 'ball':

                self.s *= 2
                self.s = self.s / np.sqrt(sum(self.s ** 2)) * np.sqrt(
                    self.I_zones.r) * np.random.random() ** (1 / self.n_obs)
                self.s += self.I_zones.center

            else:
                if self.id == 10 or self.id == 7:
                    idx = np.random.randint(self.n_obs)
                    self.s[idx] = np.random.randint(2) - 0.5
                self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        # if self.id==4:
        #     self.s[2:]=np.clip(self.s[2:],-0.2,0.2)

        # if self.id == 6:
        #
        #     self.s[3:] = np.clip(self.s[3:], -0.1, 0.1)

        return self.s

    def sample_unsafe(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
        if self.U_zones.shape == 'ball':
            s *= 2
            s = s / np.sqrt(sum(s ** 2)) * np.sqrt(self.U_zones.r)
            s += self.U_zones.center

        else:
            idx = np.random.randint(self.n_obs)
            is_up = np.random.randint(2)
            s[idx] = 0.5 if is_up else -0.5
            s = s * (self.U_zones.up - self.U_zones.low) + self.U_zones.center

        return s

    def sample_domain(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
        if self.D_zones.shape == 'ball':
            s *= 2
            s = s / np.sqrt(sum(s ** 2)) * np.sqrt(self.D_zones.r) * np.random.random() ^ (1 / self.n_obs)
            s += self.D_zones.center

        else:
            s = s * (self.D_zones.up - self.D_zones.low) + self.D_zones.center

        return s

    def sample_goal(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
        if self.G_zones.shape == 'ball':
            s *= 2
            s = s / np.sqrt(sum(s ** 2)) * np.sqrt(self.G_zones.r)

            s += self.G_zones.center
        else:
            idx = np.random.randint(self.n_obs)
            is_up = np.random.randint(2)
            s[idx] = 0.5 if is_up else -0.5
            s = s * (self.G_zones.up - self.G_zones.low) + self.G_zones.center

        return s

    def sample_goal_outer(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
        if self.G_zones.shape == 'ball':

            s *= 2
            s = s / np.sqrt(sum(s ** 2)) * np.sqrt(self.I_zones.r)
            r_sample = np.random.random() * self.I_zones.r
            if r_sample < self.G_zones.r:
                r_sample += self.G_zones.r
            s *= r_sample
            s += self.G_zones.center
        else:
            idx = np.random.randint(self.n_obs)
            is_up = np.random.randint(2)
            s[idx] = 0.5 if is_up else -0.5
            s = s * (self.G_zones.up - self.G_zones.low) + self.G_zones.center
        return s

    def step(self, u):

        self.ds = np.array([F(self.s, u) for F in self.f])
        # print('s:',self.s,'\n','ds:',self.ds)
        last_s = self.s
        self.s = self.s + self.ds * self.dt
        # print(self.s,self.ds)
        done = False
        if self.D_zones.shape == 'box':
            if self.id == 0:
                done = bool(self.s[0] < self.D_zones.low[0]
                            or self.s[0] > self.D_zones.up[0]
                            or self.s[2] < self.D_zones.low[2]
                            or self.s[2] > self.D_zones.up[2])
            else:
                for i in range(self.n_obs):
                    if self.s[i] < self.D_zones.low[i] or self.s[i] > self.D_zones.up[i]:
                        done = True
                        pass
            # self.s = np.array(
            #     [min(max(self.s[i], self.D_zones.low[i]), self.D_zones.up[i]) for i in range(self.n_obs)]
            # )
            pass

        else:
            t = np.sqrt(self.D_zones.r / sum(self.s ** 2))
            if t < 1:
                self.s = self.s * t
                done = True

            # if done:
            #     print('done!,s=',self.s)
        # reward = sum((last_s - self.G_zones.center) ** 2) - sum((self.s - self.G_zones.center) ** 2)

        if self.id == 0:
            x, x_dot, theta, theta_dot = self.s
            r1 = (self.D_zones.up[0] - abs(x)) / self.D_zones.up[0] - 0.8
            r2 = (self.D_zones.up[2] - abs(theta)) / self.D_zones.up[2] - 0.5
            reward = r1 + r2

        else:
            # reward = sum([((self.G_zones.up[i] - self.G_zones.low[i]) / 2 - abs(self.s[i] - self.G_zones.center[i])) / (
            #             self.D_zones.up[i] - self.D_zones.low[i]) for i in range(self.n_obs)]) / self.n_obs
            if self.G_zones.shape == 'box':
                if self.id == 15:
                    reward = -np.sqrt(sum((self.s[:2] - self.G_zones.center[:2]) ** 2)) + np.sqrt(
                        sum((self.G_zones.up[:2] - self.G_zones.low[:2]) ** 2 / 4))
                elif self.id == 9:
                    # gass = np.exp(-0.5 * sum(
                    #     [(self.s[i] - self.G_zones.center[i]) ** 2 / (self.G_zones.up[i] - self.G_zones.low[i]) ** 2
                    #      for i in range(self.n_obs)]))
                    # print(sum(
                    #     [(self.s[i] - self.G_zones.center[i]) ** 2 / (self.G_zones.up[i] - self.G_zones.low[i]) ** 2
                    #      for i in range(self.n_obs)]))
                    # print('gass:', gass)
                    # os.system("pause")
                    # reward = gass
                    reward = -np.sqrt(sum((self.s - self.G_zones.center) ** 2)) + np.sqrt(
                        sum((self.G_zones.up - self.G_zones.low) ** 2))
                else:
                    reward = -np.sqrt(sum((self.s - self.G_zones.center) ** 2)) + np.sqrt(
                        sum((self.G_zones.up - self.G_zones.low) ** 2 / 4))

            else:
                if self.id != 5:
                    if self.id == 4:
                        reward = -np.sqrt(sum((self.s[:2] - self.G_zones.center[:2]) ** 2)) + np.sqrt(
                            self.G_zones.r)
                    else:
                        reward = -np.sqrt(sum((self.s - self.G_zones.center) ** 2)) + np.sqrt(
                            self.G_zones.r)
                else:
                    reward = sum((last_s - self.G_zones.center) ** 2) - sum((self.s - self.G_zones.center) ** 2)
            # if self.U_zones.shape=='box':
            #     reward+=np.sqrt(sum((self.s[:2]-self.U_zones.center[:2])**2))-np.sqrt(sum((self.U_zones.up[:2]-self.U_zones.low[:2])**2))

        if self.id == 2:
            reward /= 10

        if self.id >= 3:
            if self.U_zones.shape == 'box':

                Unsafe = sum([self.U_zones.low[i] <= self.s[i] <= self.U_zones.up[i] for i in range(self.n_obs)])

                # if (Unsafe != self.n_obs) ^ self.U_zones.inner:
                #     done = True
                #     pass
                if self.id == 9:
                    gass = np.exp(-sum(
                        [(self.s[i] - self.U_zones.center[i]) ** 2 / (self.U_zones.up[i] - self.U_zones.low[i]) ** 2 for
                         i in range(self.n_obs)]))
                    reward -= gass / 3
                else:
                    if self.id != 7:
                        reward -= np.sqrt(sum((self.s - self.U_zones.center) ** 2)) * 1
                    # if Unsafe==self.n_obs:
                    #     print(self.s,gass)
            elif self.U_zones.shape == 'ball':
                if self.U_zones.inner == False:
                    Unsafe = sum((self.s - self.U_zones.center) ** 2) > self.U_zones.r
                else:
                    Unsafe = sum((self.s - self.U_zones.center) ** 2) < self.U_zones.r
                if Unsafe:
                    # if self.id==5 or self.id==4:
                    #     reward-=0.5
                    # else:
                    #     reward -= 2
                    done = True
                    pass

        # elif self.U_zones.shape=='ball':
        #     pass

        # reward=sum([(self.D_zones.up[i]-abs(self.s[i]))/self.D_zones.up[i]-0.9 for i in range(self.n_obs)])/self.n_obs
        # reward = -sum(self.s**2)/30+0.05 #np.clip(self.get_sign(u),-1,1)
        # ceter=np.array([0,-0.5,0,0,-0.5,0])
        # reward = (sum((last_s[3:]) ** 2) - sum((self.s[3:]) ** 2))*100  # - 0.002 * sum((self.s-ceter) ** 2)
        # print(reward)
        # if done:print('done:',self.s)
        reward = self.I_zones.r - sum((self.s - self.I_zones.center) ** 2)
        if self.id == 26:
            reward = -self.U_zones.r + sum((self.s - self.U_zones.center) ** 2)
        return self.s, reward, done, reward


g = 9.8
pi = np.pi
m = 0.1
l = 0.5
mt = 1.1
from numpy import sin, cos, tan, log1p


class F:
    def __init__(self):
        pass

    def f1(self, x):
        # 1/(1+sinx^2):
        return -9.49845082e-01 * x ** 2 + 9.19717026e-01 * x ** 4 - 4.06137871e-01 * x ** 6 + 0.99899106

    def f2(self, x):
        # sinx/(1+sinx^2):
        return 9.78842244e-01 * x - 8.87441593e-01 * x ** 3 + 4.35351792e-01 * x ** 5

    def f3(self, x):
        # sin(x) * cos(x) / (1 + sin(x) ** 2): \
        return 9.70088125e-01 * x - 1.27188818 * x ** 3 + 6.16181488e-01 * x ** 5

    def f4(self, x):
        # cosx / (1 + sinx ^ 2):
        return -1.42907660e+00 * x ** 2 + 1.29010139e+00 * x ** 4 - 5.75414531e-01 * x ** 6 + 0.99857329

    def f5(self, x):
        # sinx 0.005621 * x ** 5 - 0.1551 * x ** 3 + 0.9875 * x
        return 9.87855464e-01 * x - 1.55267355e-01 * x ** 3 + 5.64266597e-03 * x ** 5

    def f6(self, x):
        # cosx
        return -4.99998744e-01 * x ** 2 + 4.16558586e-02 * x ** 4 - 1.35953076e-03 * x ** 6 + 0.99999998


fun = F()


def get_Env(id):
    examples = {
        1: Example(  # useful C1
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('box', low=[-3, -3], up=[-1, -1]),
            G_zones=Zones('box', low=[-3, -3], up=[-1, -1]),
            U_zones=Zones('box', low=[2, 1], up=[4, 3]),
            f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] + u[0],
               lambda x, u: -2 * x[1] - x[0] ** 2 + u[0]],
            u=1,
            path='C1/model',
            dense=4,
            units=20,
            activation='relu',
            id=1,
            k=50
        ),
        2: Example(  # useful C2
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
            I_zones=Zones('box', low=[-1, 1], up=[-0.9, 1.1]),
            G_zones=Zones('box', low=[-1, 1], up=[-0.9, 1.1]),
            U_zones=Zones('box', low=[-2.75, -2.25], up=[-1.75, -1.25]),
            f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u[0],
               lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7)],
            u=1,
            path='C3/model',
            dense=4,
            units=20,
            activation='sigmoid',
            id=2,
            k=50
        ),

        # C3
        3: Example(  # useful C3
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[-0.1] * 2, up=[0] * 2),
            G_zones=Zones('ball', center=[1, 1], r=0.25),
            U_zones=Zones('box', low=[1.2, -0.1], up=[1.3, 0.1]),
            f=[
                lambda x, u: x[1],
                lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u[0],
            ],
            u=1,
            path='C4/model',
            dense=4,
            units=20,
            activation='relu',
            id=3,
            k=50
        ),

        4: Example(  # useful C4
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
            I_zones=Zones('box', low=[0.5, 0.5], up=[1.5, 1.5]),
            G_zones=Zones('ball', center=[1, 1], r=0.25),
            U_zones=Zones('box', low=[-1.6, -1.6], up=[-0.4, -0.4]),
            f=[lambda x, u: x[1],
               lambda x, u: -0.5 * x[0] ** 2 - x[1] + u[0],
               ],
            u=1,
            path='C4/model',
            dense=4,
            units=20,
            activation='relu',
            id=4,
            k=50
        ),
        5: Example(  # useful C5
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[1, -0.5], up=[2, 0.5]),
            G_zones=Zones('ball', center=[1, 1], r=0.25),
            U_zones=Zones('box', low=[-1.4, -1.4], up=[-0.6, -0.6]),
            f=[lambda x, u: x[1] + u[0],
               lambda x, u: -x[0] + (1 / 3) * x[0] ** 3 - x[1],
               ],
            u=1,
            path='C5/model',
            dense=4,
            units=20,
            activation='relu',
            id=5,
            k=50
        ),
        6: Example(  # useful C6
            n_obs=3,
            u_dim=1,
            D_zones=Zones(shape='box', low=[-0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2]),
            I_zones=Zones(shape='box', low=[-0.1, -0.1, -0.1], up=[0.1, 0.1, 0.1]),
            G_zones=Zones(shape='box', low=[0.18, 0.18, 0.18], up=[0.2, 0.2, 0.2]),
            U_zones=Zones(shape='box', low=[-0.18, -0.18, -0.18], up=[-0.15, -0.15, -0.15]),
            f=[lambda x, u: -x[1] + u[0],
               lambda x, u: -x[2],
               lambda x, u: -x[0] - 2 * x[1] - x[2] + x[0] ** 3,  # --+
               ],
            u=1,
            path='C6/model',
            dense=5,
            units=30,
            activation='relu',
            id=6,
            k=50,
        ),
        7: Example(  # useful C7
            n_obs=3,
            u_dim=1,
            D_zones=Zones(shape='box', low=[-4] * 3, up=[4] * 3),
            I_zones=Zones(shape='box', low=[-1] * 3, up=[1] * 3),
            G_zones=Zones(shape='box', low=[-0.4, -0.4, -0.4], up=[0.4, 0.4, 0.4]),
            U_zones=Zones(shape='box', low=[2] * 3, up=[3] * 3),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u[0],  ##--+
               ],
            u=3,
            path='C7/model',
            dense=5,
            units=30,
            activation='relu',
            id=7,
            k=50,
        ),
        8: Example(  # useful C8
            n_obs=4,
            u_dim=1,
            D_zones=Zones('box', low=[-4] * 4, up=[4] * 4),
            I_zones=Zones('box', low=[-0.2] * 4, up=[0.2, 0.2, 0.2, 0.2]),
            G_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2, 0.2]),
            U_zones=Zones('box', low=[-3] * 4, up=[-1] * 4),
            f=[lambda x, u: -x[0] - x[3] + u[0],
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u[0],
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
            u=1,
            path='C8/model',
            dense=5,
            units=30,
            activation='relu',
            id=8,
            k=100,
        ),
        9: Example(  # useful C9
            n_obs=5,
            u_dim=1,
            D_zones=Zones('box', low=[-3, -3, -3, -3, -3], up=[3, 3, 3, 3, 3]),
            I_zones=Zones('box', low=[0.5] * 5, up=[1.5] * 5),
            G_zones=Zones('ball', center=[1, 1, 1, 1, 1], r=0.25),
            U_zones=Zones('box', low=[-2.6] * 5, up=[-1.4] * 5),
            f=[
                lambda x, u: -0.1 * x[0] ** 2 - 0.4 * x[0] * x[3] - x[0] + x[1] + 3 * x[2] + 0.5 * x[3],
                lambda x, u: x[1] ** 2 - 0.5 * x[1] * x[4] + x[0] + x[2],
                lambda x, u: 0.5 * x[2] ** 2 + x[0] - x[1] + 2 * x[2] + 0.1 * x[3] - 0.5 * x[4],
                lambda x, u: x[1] + 2 * x[2] + 0.1 * x[3] - 0.2 * x[4],
                lambda x, u: x[2] - 0.1 * x[3] + u[0]
            ],
            u=5,
            path='C9/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            k=200
        ),
        10: Example(  # useful C10
            n_obs=6,
            u_dim=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[1] * 6, up=[2] * 6),
            U_zones=Zones('box', low=[-1] * 6, up=[-0.5] * 6),
            G_zones=Zones('ball', center=[0] * 6, r=0.1 ** 2),
            f=[
                lambda x, u: x[0] * x[2],
                lambda x, u: x[0] * x[4],
                lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
                lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
                lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
                lambda x, u: 2 * x[1] * x[4] + u[0]
            ],
            u=3,
            path='C10/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            k=50  # 500
        ),

        # C11
        11: Example(  # useful C11
            n_obs=6,
            u_dim=1,
            D_zones=Zones('box', low=[0] * 6, up=[10] * 6),
            I_zones=Zones('box', low=[3] * 6, up=[3.1] * 6),
            U_zones=Zones('box', low=[
                4, 4.1, 4.2, 4.3, 4.4, 4.5,
            ], up=[
                4.1, 4.2, 4.3, 4.4, 4.5, 4.6,
            ]),
            G_zones=Zones('ball', center=[0] * 6, r=0.1 ** 2),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u[0],
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
               ],
            u=3,
            path='C11/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=50  # 500
        ),
        # C12
        12: Example(  # useful C12
            n_obs=7,
            u_dim=1,
            D_zones=Zones('box', low=[-2] * 7, up=[2] * 7),
            I_zones=Zones('box', low=[0.99] * 7, up=[1.01] * 7),
            U_zones=Zones('box', low=[1.8] * 7, up=[2] * 7),
            G_zones=Zones('ball', center=[0] * 9, r=0.1 ** 2),
            f=[lambda x, u: -0.4 * x[0] + 5 * x[2] * x[3],
               lambda x, u: 0.4 * x[0] - x[1],
               lambda x, u: x[1] - 5 * x[2] * x[3],
               lambda x, u: 5 * x[4] * x[5] - 5 * x[2] * x[3],
               lambda x, u: -5 * x[4] * x[5] + 5 * x[2] * x[3],
               lambda x, u: 0.5 * x[6] - 5 * x[4] * x[5],
               lambda x, u: -0.5 * x[6] + u[0],
               ],
            u=3,
            path='C12/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=1
        ),
        13: Example(  # useful C13
            n_obs=9,
            u_dim=1,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.99] * 9, up=[1.01] * 9),
            U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
            G_zones=Zones('ball', center=[0] * 9, r=0.1 ** 2),
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
            u=3,
            path='C13/model',
            dense=5,
            units=30,
            activation='relu',
            id=13,
            k=1
        ),
        14: Example(  # useful C14
            n_obs=12,
            u_dim=1,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[
                0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5
            ], up=[
                0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, -0.5, 1.5, 1.5, -0.5, 1.5
            ]),
            G_zones=Zones('ball', center=[0] * 12, r=0.1 ** 2),
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

            u=3,
            path='C14/model',
            dense=5,
            units=30,
            activation='relu',
            id=14,
            k=1,
        ),

    }

    return Env(examples[id])
