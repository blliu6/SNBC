import re
# import gurobipy as gp
import numpy as np
import sympy as sp
import torch
from scipy.optimize import minimize, NonlinearConstraint
from benchmarks.Exampler_B import Example, get_example_by_name
from utils.Config_B import CegisConfig


def split_bounds(bounds, n):
    """
    Divide an n-dimensional cuboid into 2^n small cuboids, and output the upper and lower bounds of each small cuboid.

    parameter: bounds: An array of shape (n, 2), representing the upper and lower bounds of each dimension of an
    n-dimensional cuboid.

    return:
        An array with a shape of (2^n, n, 2), representing the upper and lower bounds of the divided 2^n small cuboids.
    """

    if n == bounds.shape[0]:
        return bounds.reshape((-1, *bounds.shape))
    else:
        # Take the middle position of the upper and lower bounds of the current dimension as the split point,
        # and divide the cuboid into two small cuboids on the left and right.
        if n > 5 and np.random.random() > 0.5:
            subbounds = split_bounds(bounds, n + 1)
        else:
            mid = (bounds[n, 0] + bounds[n, 1]) / 2
            left_bounds = bounds.copy()
            left_bounds[n, 1] = mid
            right_bounds = bounds.copy()
            right_bounds[n, 0] = mid
            # Recursively divide the left and right small cuboids.
            left_subbounds = split_bounds(left_bounds, n + 1)
            right_subbounds = split_bounds(right_bounds, n + 1)
            # Merge the upper and lower bounds of the left and right small cuboids into an array.
            subbounds = np.concatenate([left_subbounds, right_subbounds])

        return subbounds


class CounterExampleFinder():
    def __init__(self, example: Example, config: CegisConfig):
        self.n = example.n
        self.u = example.u
        self.I_zones = example.I_zones
        self.U_zones = example.U_zones
        self.D_zones = example.D_zones
        self.f = example.f
        self.choice = config.CHOICE
        self.eps = 0.05
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(self.n)])
        self.split_D_zones = split_bounds(np.array(self.D_zones), 0)
        self.config = config

    def find_init(self, B, sample_nums=10):
        samples = []
        satisfy = True

        if isinstance(B, str):
            B = sp.sympify(B)

        if self.choice[0] == 1:
            B_str = str(B)
            B_str = '-(' + B_str + ')'
            res, x_values, status = self.get_min_value(B_str, self.I_zones)
            x_values = np.array(x_values)
            if -res < 0 or status == False:
                pass
            else:
                samples.append(x_values)
                print('init_cex:', samples)
                samples.extend(self.generate_sample(x_values, sample_nums - 1))
                satisfy = False
        elif self.choice[0] == 0:
            opt = sp.lambdify(self.x, B)
            # Set an initial guess.
            x0 = np.zeros(shape=self.n)
            res = minimize(fun=lambda x: -opt(*x), x0=x0, bounds=self.I_zones)
            if -res.fun < 0:
                pass
            else:
                samples.append(res.x)
                print('init_cex:', samples)
                samples.extend(self.generate_sample(res.x, sample_nums - 1))
                satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(B, x, 'init')])

        return samples, satisfy

    def find_unsafe(self, B, sample_nums=10):
        samples = []
        satisfy = True

        if isinstance(B, str):
            B = sp.sympify(B)

        if self.choice[1] == 1:
            B_str = str(B)
            res, x_values, status = self.get_min_value(B_str, self.U_zones)
            x_values = np.array(x_values)
            if res > 0 or status == False:
                pass
            else:
                samples.append(x_values)
                print('unsafe_cex:', samples)
                samples.extend(self.generate_sample(x_values, sample_nums - 1))
                satisfy = False
        elif self.choice[1] == 0:
            opt = sp.lambdify(self.x, B)
            x0 = np.zeros(shape=(self.n))
            res = minimize(fun=lambda x: opt(*x), x0=x0, bounds=self.U_zones)

            if res.fun > 0:
                pass
            else:
                samples.append(res.x)
                samples.extend(self.generate_sample(res.x, sample_nums - 1))
                satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(B, x, 'unsafe')])

        return samples, satisfy

    def find_diff(self, B, sample_nums=10):
        samples = []
        satisfy = True
        count = 0

        if isinstance(B, str):
            B = sp.sympify(B)

        if self.choice[2] == 1:
            B_str = str(B)
            x = self.x
            DB = sum([sp.diff(B, x[i]) * self.f[i](x) for i in range(self.n)])
            DB = sp.expand(DB)
            DB_str = str(DB)
            DB_str = '-(' + DB_str + ')'
            # The condition of B(x)==0 is relaxed to find counterexamples,
            # and the existence of counterexamples does not mean that sos must not be satisfied.
            margin = 0.00
            bounds = [np.array(self.D_zones)]
            if self.config.SPLIT_D:  # todo : Parallel Computing
                bounds = self.split_D_zones

            for bound in bounds:
                res, x_values, status = self.get_min_value(DB_str, bound, B_x=B_str, margin=margin)
                x_values = np.array(x_values)

                if -res < 0 or status == False:
                    pass
                else:
                    samples.append(x_values)
                    samples.extend(self.generate_sample(x_values, sample_nums - 1))
                    satisfy = False
                    count += 1
        elif self.choice[2] == 0:
            opt = sp.lambdify(self.x, B)
            x = self.x
            DB = sum([sp.diff(B, x[i]) * self.f[i](x, [self.u]) for i in range(self.n)])
            optDB = sp.lambdify(x, DB)
            # The condition of B(x)==0 is relaxed to find counterexamples,
            # and the existence of counterexamples does not mean that sos must not be satisfied.
            margin = 0.00
            constraint = NonlinearConstraint(lambda x: opt(*x), -margin, margin)
            bounds = [np.array(self.D_zones)]
            if self.config.SPLIT_D:  # todo : Parallel Computing
                bounds = self.split_D_zones

            for bound in bounds:
                x0 = (bound.T[0] + bound.T[1]) / 2
                res = minimize(fun=lambda x: -optDB(*x), x0=x0, bounds=bound, constraints=constraint)
                if -res.fun < 0:
                    pass
                else:
                    samples.append(res.x)
                    print('Lie_cex:', samples)
                    samples.extend(self.generate_sample(res.x, sample_nums - 1))
                    satisfy = False
                    count += 1
        print('Lie derivative finds counterexamples on {} small regions!'.format(count))
        # samples = np.array([x for x in samples if self.is_counter_example(B, x, 'diff')])
        # This is a bit time consuming, it can be masked if necessary.

        return samples, satisfy

    def generate_sample(self, center, sample_nums=10):
        result = []
        for i in range(sample_nums):
            rd = (np.random.random(self.n) - 0.5) * self.eps
            rd = rd + center
            result.append(rd)
        return result

    def is_counter_example(self, B, x, condition: str) -> bool:
        if condition not in ['init', 'unsafe', 'diff']:
            raise ValueError(f'{condition} is not in validation condition!')
        d = {'x{}'.format(i + 1): x[i] for i in range(self.n)}
        b_numerical = B.subs(d)
        dot_b = sum([sp.diff(B, self.x[i]) * self.f[i](x) for i in range(self.n)])
        dot_b_numerical = dot_b.subs(d)
        if condition == 'init' and b_numerical > 0:
            return True
        if condition == 'unsafe' and b_numerical < 0:
            return True
        if condition == 'diff' and dot_b_numerical > 0:
            return True
        return False

    def get_counter_example(self, B):
        samples = []
        S, satisfy1 = self.find_init(B, 100)
        samples.append(torch.Tensor(S))
        S, satisfy2 = self.find_unsafe(B, 100)
        samples.append(torch.Tensor(S))
        S, satisfy3 = self.find_diff(B, 100 if self.config.SPLIT_D else 100)
        samples.append(torch.Tensor(np.array(S)))

        return samples, (satisfy1 & satisfy2 & satisfy3)

