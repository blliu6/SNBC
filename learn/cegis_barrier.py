import torch
import timeit
import numpy as np

import benchmarks.Exampler_B
from utils.Config_B import CegisConfig
from learn.net_B import Learner
from verify.CounterExampleFind_B import CounterExampleFinder
from verify.SosVerify_B import SosValidator_B
from learn.save_result import SaveResult


class Cegis:
    def __init__(self, config: CegisConfig):
        n = config.EXAMPLE.n
        self.ex = config.EXAMPLE
        self.n = n
        self.f = config.EXAMPLE.f
        self.I_zones = np.array(config.EXAMPLE.I_zones, dtype=np.float32).T
        self.U_zones = np.array(config.EXAMPLE.U_zones, dtype=np.float32).T
        self.D_zones = np.array(config.EXAMPLE.D_zones, dtype=np.float32).T
        self.batch_size = config.BATCH_SIZE
        self.learning_rate = config.LEARNING_RATE
        self.Learner = Learner(config)

        self.optimizer = config.OPT(self.Learner.net.parameters(), lr=self.learning_rate)
        self.CounterExampleFinder = CounterExampleFinder(config.EXAMPLE, config)

        self.max_cegis_iter = config.LEARNING_LOOPS
        self.DEG = config.DEG
        self.R_b = config.R_b

        self._assert_state()
        self._result = None
        self.config = config

    def solve(self):
        # check_mosek
        import mosek

        with mosek.Env() as env:
            with env.Task() as task:
                task.optimize()
                print('Mosek can be used normally.')

        S_i, S_u, S_d, Sdot_i, Sdot_u, Sdot_d = self.generate_data()

        S, Sdot = [S_i, S_u, S_d], [Sdot_i, Sdot_u, Sdot_d]

        # the CEGIS loop
        deg = self.DEG
        t_learn = 0
        t_cex = 0
        t_sos = 0
        for i in range(self.max_cegis_iter):
            t1 = timeit.default_timer()
            self.Learner.learn(self.optimizer, S, Sdot)
            t2 = timeit.default_timer()
            t_learn += t2 - t1
            B = self.Learner.net.get_barrier()

            print(f'iter: {i + 1} \nB = {B}')
            t3 = timeit.default_timer()
            Sos_Validator = SosValidator_B(self.ex, B)
            if Sos_Validator.SolveAll(deg=deg):
                print('SOS verification passed!')
                t4 = timeit.default_timer()
                t_sos += t4 - t3
                saver = SaveResult(self.config, [t_learn, t_cex, t_sos, i + 1], B, self.Learner.net)
                saver.save_all()
                break
            t4 = timeit.default_timer()
            t_sos += t4 - t3
            # In the negative example of Lie derivative, the condition of B(x)==0 is relaxed to |B(x)|<=margin
            # to find a negative example, so the existence of a negative example does not mean that sos must
            # not be satisfied
            t5 = timeit.default_timer()
            samples, satisfy = self.CounterExampleFinder.get_counter_example(B)
            t6 = timeit.default_timer()
            t_cex += t6 - t5
            if satisfy:
                print('No counterexamples were found!')

            # if satisfy:
            #     # If no counterexample is found, but SOS fails, it may be that the number of multipliers is too low.
            #     deg[3] += 2

            S, Sdot = self.add_ces_to_data(S, Sdot, samples)
            print('-' * 200)
        print('Total learning time:{}'.format(t_learn))
        print('Total counter-examples generating time:{}'.format(t_cex))
        print('Total sos verifying time:{}'.format(t_sos))

    def add_ces_to_data(self, S, Sdot, ces):
        """
        :param S: torch tensor
        :param Sdot: torch tensor
        :param ces: list of ctx
        :return:
                S: torch tensor, added new ctx
                Sdot torch tensor, added  f(new_ctx)
        """
        assert len(ces) == 3

        for idx in range(3):
            if len(ces[idx]) != 0:
                print(f'Add {len(ces[idx])} counterexamples!')
                S[idx] = torch.cat([S[idx], ces[idx]], dim=0).detach()
                dot_ces = self.x2dotx(ces[idx])
                Sdot[idx] = torch.cat([Sdot[idx], dot_ces], dim=0).detach()

        return S, Sdot

    def generate_data(self):

        I_len = self.I_zones[1] - self.I_zones[0]
        U_len = self.U_zones[1] - self.U_zones[0]
        D_len = self.D_zones[1] - self.D_zones[0]

        # Let more sample points fall on the boundary.
        times = 1 / (1 - self.R_b)
        S_i = torch.clamp((torch.rand([self.batch_size, self.n]) - 0.5) * times, -0.5, 0.5) + 0.5
        S_u = torch.clamp((torch.rand([self.batch_size, self.n]) - 0.5) * times, -0.5, 0.5) + 0.5
        S_d = torch.clamp((torch.rand([self.batch_size, self.n]) - 0.5) * times, -0.5, 0.5) + 0.5

        S_i = S_i * I_len + self.I_zones[0]
        S_u = S_u * U_len + self.U_zones[0]
        S_d = S_d * D_len + self.D_zones[0]

        Sdot_i, Sdot_d, Sdot_u = self.x2dotx(S_i), self.x2dotx(S_d), self.x2dotx(S_u)

        return S_i, S_u, S_d, Sdot_i, Sdot_u, Sdot_d

    def x2dotx(self, X):
        f_x = []
        for x in X:
            f_x.append([self.f[i](x, [self.ex.f_u(*x)]) for i in range(self.n)])

        return torch.Tensor(f_x)

    def _assert_state(self):
        assert self.batch_size > 0
        assert self.learning_rate > 0


if __name__ == '__main__':
    pass
