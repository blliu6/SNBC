from RL_train.train import RlTrain
from RL_train.PolynomialFit import PolFit
import numpy as np


def main():
    identity = 22
    trainer = RlTrain(identity, episodes=20, max_steps=1000)
    trainer.learn()

    fiter = PolFit(identity, iter=100, max_step=1000)
    p = fiter.fit()


if __name__ == '__main__':
    np.random.seed(2023)
    main()
