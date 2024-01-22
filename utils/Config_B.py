import torch


class CegisConfig():
    N_HIDDEN_NEURONS = [10]
    EXAMPLE = None
    ACTIVATION = ['SQUARE']
    MULTIPLICATOR = False
    BATCH_SIZE = 500
    LEARNING_RATE = 0.01
    LOSS_WEIGHT = (1, 1, 1)
    BIAS = True
    SPLIT_D = False
    MARGIN = 0.5
    DEG = [2, 2, 2, 2]
    MULTIPLICATOR_NET = []
    MULTIPLICATOR_ACT = []
    OPT = torch.optim.AdamW
    R_b = 0.4
    LEARNING_LOOPS = 100
    CHOICE = [0, 0, 0]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if len(self.MULTIPLICATOR_NET) > 0:
            assert len(self.MULTIPLICATOR_ACT) == len(self.MULTIPLICATOR_NET) - 1
