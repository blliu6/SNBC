import sys
sys.path.append("SynNBC-main")
from learn.cegis_barrier import Cegis
from utils.Config_B import CegisConfig
import timeit
import torch
from benchmarks.Exampler_B import get_example_by_name, get_example_by_env
from plots.plot_barriers import plot_benchmark2d
from RL_learn.Env import get_Env
from RL_learn.train import RlTrain
from RL_learn.PolynomialFit import PolFit
import numpy as np
import sympy as sp

def main():

    identity = 6
    trainer = RlTrain(identity, episodes=30, max_steps=1000)
    trainer.learn()

    fiter = PolFit(identity, iter=100, max_step=1000)
    p = fiter.fit()

    activations = ['MUL']  # Only "SQUARE","SKIP","MUL" are optional.
    hidden_neurons = [10] * len(activations)

    example = get_example_by_env(get_Env(identity))
    x = sp.symbols([f'x{i + 1}' for i in range(example.n)])
    f_u = sp.lambdify(x, p)
    example.f_u = f_u
    example.u = p

    start = timeit.default_timer()
    opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "MULTIPLICATOR": True,  # Whether to use multiplier.
        "MULTIPLICATOR_NET": [5,1],  # The number of nodes in each layer of the multiplier network;
        # if set to empty, the multiplier is a trainable constant.
        "MULTIPLICATOR_ACT": ['LINEAR'],
        # The activation function of each layer of the multiplier network;
        # since the last layer does not require an activation function, the number is one less than MULTIPLICATOR_NET.
        "BATCH_SIZE": 1000,
        "LEARNING_RATE": 0.1,
        "MARGIN": 2,
        "LOSS_WEIGHT": (1.0, 1.0, 1.0),  # # They are the weights of init loss, unsafe loss, and diffB loss.
        "SPLIT_D": True,  # Indicates whether to divide the region into 2^n small regions
        # when looking for negative examples, and each small region looks for negative examples separately.
        "DEG": [2, 2, 2, 1],  # Respectively represent the times of init, unsafe, diffB,
        # and unconstrained multipliers when verifying sos.
        "R_b": 0.6,
        "LEARNING_LOOPS": 100,
        "CHOICE": [0, 0, 0]  # For finding the negative example, whether to use the minimize function or the gurobi
        # solver to find the most value, 0 means to use the minimize function, 1 means to use the gurobi solver; the
        # three items correspond to init, unsafe, and diffB to find the most value. (note: the gurobi solver does not
        # supports three or more objective function optimizations.)
    }
    Config = CegisConfig(**opts)
    c = Cegis(Config)
    c.generate_data()
    c.solve()
    end = timeit.default_timer()
    print('Elapsed Time: {}'.format(end - start))
    plot_benchmark2d(example, c.Learner.net.get_barrier())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
