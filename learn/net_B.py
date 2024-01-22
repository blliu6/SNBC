import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from utils.Config_B import CegisConfig


class Net(nn.Module):
    def __init__(self, config: CegisConfig):
        super(Net, self).__init__()
        self.config = config
        self.input_size = input_size = config.EXAMPLE.n  # The number of nodes in the input layer is the number of var
        hiddens = config.N_HIDDEN_NEURONS
        activate = config.ACTIVATION
        self.acts = config.ACTIVATION
        self.bias = bias = config.BIAS
        self.layers1 = []  # backbone network,such as square
        self.layers2 = []  # used for skip, mul and other networks
        self.multiplicators1 = []  # multiplier network
        self.multiplicators2 = []

        if config.MULTIPLICATOR:
            if len(config.MULTIPLICATOR_NET) == 0:
                scalar = torch.nn.Parameter(torch.randn(1))
                self.register_parameter('scalar', scalar)
                self.multiplicators1.append(scalar)
            else:
                assert config.MULTIPLICATOR_NET[-1] == 1

                n_prev = config.EXAMPLE.n
                j = 1
                for n_hid, act in zip(config.MULTIPLICATOR_NET[:-1], config.MULTIPLICATOR_ACT):
                    multor1 = nn.Linear(n_prev, n_hid)
                    if act == 'SKIP':
                        multor2 = nn.Linear(input_size, n_hid, bias=bias)
                    else:
                        multor2 = nn.Linear(n_prev, n_hid, bias=bias)
                    self.register_parameter("M1" + str(j), multor1.weight)
                    self.register_parameter("M2" + str(j), multor2.weight)
                    if bias:
                        self.register_parameter("mb1" + str(j), multor1.bias)
                        self.register_parameter("mb2" + str(j), multor2.bias)
                    self.multiplicators1.append(multor1)
                    self.multiplicators2.append(multor2)
                    n_prev = n_hid
                    j = j + 1
                multor1 = nn.Linear(n_prev, 1)
                self.multiplicators1.append(multor1)
                self.register_parameter("M1" + str(j), multor1.weight)
                self.register_parameter("mb1" + str(j), multor1.bias)

        k = 1
        n_prev = config.EXAMPLE.n
        for n_hid, act in zip(hiddens, activate):
            layer1 = nn.Linear(n_prev, n_hid, bias=bias)

            if act not in ['SKIP']:
                layer2 = nn.Linear(n_prev, n_hid, bias=bias)
            else:
                layer2 = nn.Linear(input_size, n_hid, bias=bias)

            self.register_parameter("W" + str(k), layer1.weight)
            self.register_parameter("W2" + str(k), layer2.weight)
            if bias:
                self.register_parameter("b" + str(k), layer1.bias)
                self.register_parameter("b2" + str(k), layer2.bias)
            self.layers1.append(layer1)
            self.layers2.append(layer2)

            n_prev = n_hid
            k = k + 1

        # free output layer
        layer1 = nn.Linear(n_prev, 1, bias=False)
        self.register_parameter("W" + str(k), layer1.weight)
        self.layers1.append(layer1)

    def forward(self, x, xdot):
        yy = x
        if self.config.MULTIPLICATOR:
            if len(self.config.MULTIPLICATOR_NET) == 0:
                yy = x * 0 + self.multiplicators1[0]
            else:
                relu6 = nn.ReLU6()
                for idx, (mul1, mul2) in enumerate(zip(self.multiplicators1[:-1], self.multiplicators2)):
                    if self.config.MULTIPLICATOR_ACT[idx] == 'ReLU':
                        yy = relu6(mul1(yy))
                    elif self.config.MULTIPLICATOR_ACT[idx] == 'SQUARE':
                        yy = mul1(yy) ** 2
                    elif self.config.MULTIPLICATOR_ACT[idx] == 'MUL':
                        yy = mul1(yy) * mul2(yy)
                    elif self.config.MULTIPLICATOR_ACT[idx] == 'SKIP':
                        yy = mul1(yy) * mul2(x)
                    elif self.config.MULTIPLICATOR_ACT[idx] == 'LINEAR':
                        yy = mul1(yy)

                yy = self.multiplicators1[-1](yy)

        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input_size))
        for idx, (layer1, layer2) in enumerate(zip(self.layers1[:-1], self.layers2)):
            if self.acts[idx] in ['SQUARE']:
                z = layer1(y)
                y = z ** 2
                jacobian = torch.matmul(torch.matmul(2 * torch.diag_embed(z), layer1.weight), jacobian)

            elif self.acts[idx] == 'MUL':
                z1 = layer1(y)
                z2 = layer2(y)
                y = z1 * z2
                grad = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(torch.diag_embed(z2),
                                                                                        layer1.weight)
                jacobian = torch.matmul(grad, jacobian)


            elif self.acts[idx] == 'SKIP':
                z1 = layer1(y)
                z2 = layer2(x)
                y = z1 * z2
                jacobian = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(
                    torch.matmul(torch.diag_embed(z2), layer1.weight), jacobian)

        numerical_b = torch.matmul(y, self.layers1[-1].weight.T)
        jacobian = torch.matmul(self.layers1[-1].weight, jacobian)
        numerical_bdot = torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)

        # return the value of BC, the value of Lie derivative, BC, multiplier
        return numerical_b, numerical_bdot, y, yy

    def get_barrier(self):
        """
        get BC from net.
        """

        x = sp.symbols([['x{}'.format(i + 1) for i in range(self.input_size)]])
        y = x
        for idx, (layer1, layer2) in enumerate(zip(self.layers1[:-1], self.layers2)):
            if self.acts[idx] == 'SQUARE':
                w1 = layer1.weight.detach().numpy()
                b1 = layer1.bias.detach().numpy()
                z = np.dot(y, w1.T) + b1
                y = z ** 2
            elif self.acts[idx] == 'MUL':
                w1 = layer1.weight.detach().numpy()
                b1 = layer1.bias.detach().numpy()
                z1 = np.dot(y, w1.T) + b1

                w2 = layer2.weight.detach().numpy()
                b2 = layer2.bias.detach().numpy()
                z2 = np.dot(y, w2.T) + b2

                y = np.multiply(z1, z2)
            elif self.acts[idx] == 'SKIP':
                w1 = layer1.weight.detach().numpy()
                b1 = layer1.bias.detach().numpy()
                z1 = np.dot(y, w1.T) + b1

                w2 = layer2.weight.detach().numpy()
                b2 = layer2.bias.detach().numpy()
                z2 = np.dot(x, w2.T) + b2
                y = np.multiply(z1, z2)

        w1 = self.layers1[-1].weight.detach().numpy()
        y = np.dot(y, w1.T)
        y = sp.expand(y[0, 0])
        return y


class Learner(nn.Module):
    def __init__(self, config: CegisConfig):
        super(Learner, self).__init__()
        self.net = Net(config)
        self.loss_weight = config.LOSS_WEIGHT
        self.config = config

    def learn(self, optimizer, S, Sdot):
        """
        :param optimizer: torch optimiser
        :param S: tensor of data
        :param Sdot: tensor contain f(data)
        :param margin: performance threshold
        :return: --
        """

        assert (len(S) == len(Sdot))
        print('Init samples:', len(S[0]), 'Unsafe samples:', len(S[1]), 'Lie samples', len(S[2]))
        learn_loops = self.config.LEARNING_LOOPS
        margin = self.config.MARGIN
        slope = 1e-3
        relu6 = torch.nn.ReLU6()
        for t in range(learn_loops):
            optimizer.zero_grad()

            B_i, _, __, ___ = self.net(S[0], Sdot[0])
            B_u, _, __, ___ = self.net(S[1], Sdot[1])
            B_d, Bdot_d, __, yy = self.net(S[2], Sdot[2])

            B_i = B_i[:, 0]
            B_u = B_u[:, 0]
            B_d = B_d[:, 0]
            yy = yy[:, 0]
            accuracy_init = sum(B_i < -margin / 2).item() * 100 / len(S[0])
            accuracy_unsafe = sum(B_u > margin / 2).item() * 100 / len(S[1])

            loss = self.loss_weight[0] * (torch.relu(B_i + margin) - slope * relu6(-B_i - margin)).mean()
            loss = loss + self.loss_weight[1] * (torch.relu(-B_u + margin) - slope * relu6(B_u - margin)).mean()

            belt_index = torch.nonzero(torch.abs(B_d) <= 5.0)

            dB_belt = torch.index_select(Bdot_d, dim=0, index=belt_index[:, 0])
            if self.config.MULTIPLICATOR:
                dB_belt = Bdot_d - yy * B_d
                loss = loss + self.loss_weight[2] * (
                        torch.relu(dB_belt + margin) - slope * relu6(-dB_belt - margin)).mean()
            elif belt_index.nelement() != 0:
                loss = loss - self.loss_weight[2] * (relu6(-dB_belt + margin)).mean()
            if dB_belt.shape[0] > 0:
                percent_belt = 100 * (sum(dB_belt <= -margin)).item() / dB_belt.shape[0]
            else:
                percent_belt = 0

            if t % int(learn_loops / 10) == 0 or (
                    accuracy_init == 100 and percent_belt == 100 and accuracy_unsafe == 100):
                belt = ('- points in belt: {}'.format(len(belt_index))) if not self.config.MULTIPLICATOR else ''
                print(t, "- loss:", loss.item(), '- accuracy init:', accuracy_init, 'accuracy unsafe:', accuracy_unsafe,
                      "- accuracy Lie:", percent_belt, belt)

            loss.backward()
            optimizer.step()
            if (accuracy_init == 100 and percent_belt == 100 and accuracy_unsafe == 100):
                print('Average multiplier size:', torch.mean(yy))
                break

        return {}
