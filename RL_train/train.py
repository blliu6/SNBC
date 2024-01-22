from collections import deque

import numpy as np
import torch

from RL_train.DDPG import DDPG
from RL_train.Env import get_Env
from RL_train.Plot import plot


class RlTrain:
    def __init__(self, identity, episodes=300, max_steps=2000):
        self.id = identity
        self.episodes = episodes
        self.max_steps = max_steps

    def learn(self):
        m = 5
        epoch_reward = np.zeros(m, np.float32)
        epoch_len = np.zeros_like(epoch_reward)

        best = -500000000
        exploration = 1
        device = torch.device('cpu')
        env = get_Env(self.id)

        agent = DDPG(a_dim=env.u_dim, a_bound=env.u, s_dim=env.n_obs, path=env.path, units=env.units,
                     dense=env.dense, activation=env.activation, device=device)

        MAX_EPISODES = self.episodes
        MAX_EP_STEPS = self.max_steps
        dq = deque(maxlen=1)
        var = 5

        for i in range(MAX_EPISODES):
            s = env.reset()
            print('s:', s)
            X, Y, Z = [], [], []
            ep_reward = 0
            tot = 0

            while True:
                u = agent.choose_action(s)

                X.append(s[0])
                Y.append(s[1])
                if exploration == 1:
                    a_v = np.clip(np.random.normal(u, var), -env.u,
                                  env.u)
                else:
                    sigma = 0.01
                    a_v = u + np.random.randn(env.u_dim) * sigma
                    a_v = np.clip(a_v, -env.u, env.u)

                s_, r, done, info = env.step(a_v)
                agent.store_transition(s, a_v, r, s_, done)

                if agent.pointer > 1000:
                    if tot % 20 == 0:
                        if var > 0.5:
                            var *= .9995  # decay the action randomness

                    agent.learn()

                s = s_

                ep_reward += r
                if done:
                    print('Last position:', s_)
                if tot >= MAX_EP_STEPS:
                    done = True

                if done:
                    print('Episode:', i, ' Reward: %.2f' % ep_reward,
                          ('Explore: %.2f' % var), '  tot:', tot)
                    epoch_reward[i % m] = ep_reward
                    epoch_len[i % m] = tot
                    if i > m and np.average(epoch_reward) > best and np.average(
                            epoch_len) > MAX_EP_STEPS / 1.5:
                        best = np.average(epoch_reward)

                        agent.save()
                        print('best:', best)
                        plot(env, dq, i, best=True)
                    break
                tot += 1

            dq.append((X, Y))
            if i % 10 == 1:
                plot(env, dq, i)
