import time
from functools import reduce
from itertools import product
import sympy as sp
from SumOfSquares import SOSProblem
from benchmarks.Exampler_B import Example, get_example_by_name


class SosValidator_B():
    def __init__(self, example: Example, B) -> None:
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(example.n)])
        self.n = example.n
        self.Inits = example.I_zones
        self.Unsafes = example.U_zones
        self.Invs = example.D_zones
        self.f = [example.f[i](self.x, [example.u]) for i in range(self.n)]
        self.B = B
        self.var_count = 0

    def polynomial(self, deg=2):  # Generating polynomials of degree n-ary deg.
        if deg == 2 and self.n > 8:
            parameters = []
            terms = []
            poly = 0
            parameters.append(sp.symbols('parameter' + str(self.var_count)))
            self.var_count += 1
            poly += parameters[-1]
            terms.append(1)
            for i in range(self.n):
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i])
                poly += parameters[-1] * terms[-1]
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i] ** 2)
                poly += parameters[-1] * terms[-1]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    parameters.append(sp.symbols('parameter' + str(self.var_count)))
                    self.var_count += 1
                    terms.append(self.x[i] * self.x[j])
                    poly += parameters[-1] * terms[-1]
            return poly, parameters, terms
        else:
            parameters = []
            terms = []
            exponents = list(product(range(deg + 1), repeat=self.n))  # Generate all possible combinations of indices.
            exponents = [e for e in exponents if sum(e) <= deg]  # Remove items with a count greater than deg.
            poly = 0
            for e in exponents:  # Generate all items.
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(reduce(lambda a, b: a * b, [self.x[i] ** exp for i, exp in enumerate(e)]))
                poly += parameters[-1] * terms[-1]
            return poly, parameters, terms

    def SovleInit(self, deg=2):
        prob_init = SOSProblem()
        B_I = -self.B
        x = self.x
        Inits = self.Inits

        for i in range(self.n):
            Pi, parameters, terms = self.polynomial(deg)
            prob_init.add_sos_constraint(Pi, x)
            B_I = B_I + Pi * (x[i] - Inits[i][0]) * (x[i] - Inits[i][1])
        B_I = sp.expand(B_I)
        prob_init.add_sos_constraint(B_I, x)
        try:
            prob_init.solve(solver='mosek')
            return True
        except:
            return False

    def SovleUnsafe(self, deg=2):
        prob_unsafe = SOSProblem()
        B_U = self.B
        x = self.x
        Unsafes = self.Unsafes
        for i in range(self.n):
            Qi, parameters, terms = self.polynomial(deg)
            prob_unsafe.add_sos_constraint(Qi, x)
            B_U = B_U + Qi * (x[i] - Unsafes[i][0]) * (x[i] - Unsafes[i][1])
        B_U = sp.expand(B_U)
        prob_unsafe.add_sos_constraint(B_U, x)
        try:
            prob_unsafe.solve(solver='mosek')
            return True
        except:
            return False

    def SolveDiffB(self, deg=[2, 2]):
        prob_inv = SOSProblem()
        x = self.x
        Invs = self.Invs
        B = self.B
        DB = -sum([sp.diff(B, x[i]) * self.f[i] for i in range(self.n)])
        for i in range(self.n):
            Si, parameters, terms = self.polynomial(deg[0])
            prob_inv.add_sos_constraint(Si, x)
            DB = DB + Si * (x[i] - Invs[i][0]) * (x[i] - Invs[i][1])

        R1, parameters, terms = self.polynomial(deg[1])
        DB = DB - B * R1
        DB = sp.expand(DB)
        prob_inv.add_sos_constraint(DB, x)
        try:
            prob_inv.solve(solver='mosek')
            return True
        except:
            return False

    def SolveAll(self, deg=(2, 2, 2, 2)):
        assert len(deg) == 4

        Init = self.SovleInit(deg[0])
        if not Init:
            print('The initial set is not satisfied.')
            return False
        Unsafe = self.SovleUnsafe(deg[1])
        if not Unsafe:
            print('The unsafe set is not satisfied.')
            return False
        DB = self.SolveDiffB(deg[2:])
        if not DB:
            print('The Lie derivative is not satisfied.')
            return False

        return True

