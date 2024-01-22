import re
from collections import deque

import numpy as np
import sympy as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from RL_train.DDPG import DDPG
from RL_train.Env import get_Env


class PolFit:
    def __init__(self, identity, iter=200, max_step=1000):
        self.id = identity
        self.iter = iter
        self.max_step = max_step
        self.degree = 2

    def fit(self):
        X, Y = self.get_data()
        P = PolynomialFeatures(self.degree, include_bias=False)
        x = P.fit_transform(X)
        model = Ridge(alpha=0.00, fit_intercept=False)
        model.fit(x, Y)
        y = model.predict(x)

        print('socre:', model.score(x, Y))
        s = ''
        for j in range(len(model.coef_)):
            for k, v in zip(P.get_feature_names_out(), model.coef_[j]):
                k = re.sub(r' ', r'*', k)
                k = k.replace('^', '**')
                if v < 0:
                    s += f'- {-v} * {k} '
                else:
                    s += f'+ {v} * {k} '
        s = s[1:]
        x = sp.symbols([f'x{i}' for i in range(len(X[0]))])
        x_ =sp.symbols([f'x{i+1}' for i in range(len(X[0]))])
        s_f = sp.lambdify(x,s)
        ans = s_f(*x_)
        print(ans)
        return ans

    def get_data(self):
        env = get_Env(self.id)
        agent = DDPG(env.u_dim, env.n_obs, env.u, dense=env.dense, units=env.units, path=env.path,
                     is_train=False, activation=env.activation)

        X, Y = [], []
        unsafe_times = 0
        N = self.iter
        for t in range(N):
            s = env.reset()
            tot = 0
            mxlen = 100
            dq = deque(maxlen=mxlen)
            reward = 0
            while True:
                tot += 1

                if tot >= self.max_step:
                    print('1 the {} track'.format(t))
                    break
                dq.append(sum(env.s ** 2))

                if len(dq) == mxlen and np.var(dq) < 1e-6:
                    print('2 the {} track'.format(t))
                    break

                a = agent.choose_action(s)
                a = np.clip(a, -env.u, env.u)

                X.append(s)
                Y.append(a)

                s_, r, done, info = env.step(a)
                reward += r
                s = s_.copy()

                if done:
                    print('Entering an unsafe area!')
                    unsafe_times += 1
                    print(tot)
                    break

        return X, Y
