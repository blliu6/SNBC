import os
import joblib
import numpy as np
import torch
from torch import nn


class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return x ** 2


class Rational(nn.Module):
    def __init__(self, dim):
        super(Rational, self).__init__()
        self.a = nn.init.uniform_(nn.Parameter(torch.empty((3, dim))))
        self.b = nn.init.uniform_(nn.Parameter(torch.empty(1, dim)))

    def forward(self, x):
        up = self.a[0] + self.a[1] * x + self.a[2] * x ** 2
        down = 1 + (self.b * x) ** 2
        return up / down


class Actor(nn.Module):
    def __init__(self, a_dim, s_dim, a_bound, dense=3, units=20, activation='relu'):
        super(Actor, self).__init__()
        self.seq = nn.Sequential()
        func = {'relu': torch.nn.ReLU(), 'square': Square(), 'rational': Rational(units), 'sigmoid': torch.nn.Sigmoid()}
        for i in range(dense):
            self.seq.add_module('linear_{}'.format(i), nn.Linear(s_dim, units))
            self.seq.add_module('rational_{}'.format(i), func[activation])
            s_dim = units
        self.seq.add_module('linear_{}'.format(dense), nn.Linear(units, a_dim))
        self.seq.add_module('tanh', nn.Tanh())
        self.bound = a_bound

    def forward(self, x):
        return self.seq(x) * self.bound


class Critic(nn.Module):
    def __init__(self, a_dim, s_dim, dense=3, units=20):
        super(Critic, self).__init__()
        self.seq = nn.Sequential()
        s = a_dim + s_dim
        for i in range(dense):
            self.seq.add_module('linear_{}'.format(i), nn.Linear(s, units))
            self.seq.add_module('relu_{}'.format(i), nn.ReLU())
            s = units
        self.seq.add_module('linear_{}'.format(dense), nn.Linear(units, 1))

    def forward(self, x1, x2):
        # x1=x1/10
        x = torch.cat([x1, x2], dim=1)
        # print(x[0])
        return self.seq(x)


class DDPG(nn.Module):
    LR_A = 1e-5  # learning rate for actor
    LR_C = 2e-4  # learning rate for critic
    GAMMA = 0.999  # reward discount
    TAU = 0.1  # soft replacement
    MEMORY_CAPACITY = 50000
    BATCH_SIZE = 64
    learn_step_counter = 0
    pointer = 0

    def __init__(self, a_dim, s_dim, a_bound, path='Pendulum', is_train=True, dense=3, units=20, activation='relu',
                 device=torch.device('cpu')):
        super(DDPG, self).__init__()
        self.memory = torch.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1))

        self.is_load_variable = not is_train
        self.a_dim, self.s_dim = a_dim, s_dim
        self.actor = Actor(a_dim, s_dim, a_bound, dense, units, activation).to(device)
        self.critic = Critic(a_dim, s_dim, dense, units).to(device)
        self.actor_ = Actor(a_dim, s_dim, a_bound, dense, units, activation).to(device)  # replaced target parameters
        self.critic_ = Critic(a_dim, s_dim, dense, units).to(device)
        self.device = device
        for target_param, source_param in zip(self.actor_.parameters(), self.actor.parameters()):
            target_param.data.copy_(source_param.data)
            target_param.requires_grad = False  # target网络不训练
        for target_param, source_param in zip(self.critic_.parameters(), self.critic.parameters()):
            target_param.data.copy_(source_param.data)
            target_param.requires_grad = False

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.LR_A)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.LR_C)
        self.mse = nn.MSELoss()
        self.save_path = './ddpg/' + path
        if not os.path.exists('/'.join(self.save_path.split('/')[:-1])):
            os.makedirs('/'.join(self.save_path.split('/')[:-1]))
        if self.is_load_variable:
            print(self.save_path)
            st = joblib.load(self.save_path)
            self.actor.load_state_dict(st[0])
            self.actor_.load_state_dict(st[1])
            self.critic.load_state_dict(st[2])
            self.critic_.load_state_dict(st[3])
            print('Loading parameters successfully!')

    def choose_action(self, s):
        s = torch.unsqueeze(torch.Tensor(s), dim=0).to(self.device)

        return self.actor(s).cpu().detach().numpy()[0]

    def save(self):
        st = {0: self.actor.state_dict(), 1: self.actor_.state_dict(), 2: self.critic.state_dict(),
              3: self.critic_.state_dict()}
        joblib.dump(st, self.save_path)
        print('Model saved successfully!')

    def learn(self):
        self.learn_step_counter += 1

        indices = np.random.choice(min(self.MEMORY_CAPACITY, self.pointer), size=self.BATCH_SIZE)
        # print(self.pointer)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim].to(self.device)  # state
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim].to(self.device)  # action
        br = bt[:, -self.s_dim - 1 - 1: -self.s_dim - 1].to(self.device)  # reward
        bs_ = bt[:, -self.s_dim - 1:-1].to(self.device)  # next_state
        bdone = bt[:, -1:].to(self.device)  # done
        # print(ba)
        # print(bs.shape,ba.shape,br.shape,bs_.shape,bdone.shape)

        q = self.critic(self.actor(bs), bs)

        loss_a = -torch.mean(q)
        # print('qian:',loss_a)
        self.opt_actor.zero_grad()
        loss_a.backward(retain_graph=True)
        self.opt_actor.step()
        # print('hou:',-torch.mean(self.critic(self.actor(bs),bs)))

        q_ = self.critic_(self.actor_(bs_), bs_)
        # print((br+(1-bdone)*self.GAMMA*q_).shape)
        loss_c = self.mse(self.critic(ba, bs), br + (1 - bdone) * self.GAMMA * q_)
        # print('qian:',loss_c)

        self.opt_critic.zero_grad()
        # loss_c.backward(retain_graph=True)
        loss_c.backward()
        self.opt_critic.step()

        # print(q[0],q_[0])
        # loss_c=self.mse(self.critic(ba,bs),br+(1-bdone)*self.GAMMA*q_)
        # print('hou:',loss_c)
        # print('----------------------------------------' * 2)

        for target_param, source_param in zip(self.actor_.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.TAU) * target_param.data + self.TAU * source_param.data)

        for target_param, source_param in zip(self.critic_.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.TAU) * target_param.data + self.TAU * source_param.data)

    def store_transition(self, s, a, r, s_, done):

        transition = torch.FloatTensor(np.hstack((s, a, [r], s_, done)))
        # print(transition)
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
