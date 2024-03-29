# 1.Introduction

SNBC is a Python tool for synthesizing neural barrier certificate of the NN-controlled system. We provide an abstract method to obtain polynomial inclusions for NN-controllers trained by DDPG algorithm of reinforcement learning,
and explore counterexample-guided learning approach to
yield a neural barrier certificate for the closed-loop system
with the abstract controller.

* `/benchmarks`: the source code and some examples;
* `/learn`: the code of learner of barrier certificate;
* `/verify`: the code of verifier and counterexample generation;
* `/plots`: the code of plots;
* `/RL_train`: the code of training controller by reinforcement learning and polynomial P(x) abstraction;
* `/utils`: the configuration of the project.


# 2.Configuration

## 2.1 Project requirements

To install and run SNBC, you need:

* Windows Platform: `Python 3.9`;
* Linux Platform: `Python 3.9`;
* Mac OS X Platform: `Python 3.9`.

## 2.2 Installation instruction

You need install required software packages listed below and setting up a MOSEK license .

1. Download SNBC.zip, and unpack it;
2. Install the required software packages for using SNBC:

    ```python
    pip install cvxopt==1.3.0
    pip intsall matplotlib==3.5.3
    pip intsall numpy==1.23.2
    pip intsall scipy==1.9.0
    pip intsall SumOfSquares==1.2.1
    pip intsall sympy==1.11
    pip intsall torch==1.12.1
    pip install Mosek==10.0.30
    pip install picos==2.4.11
    pip install joblib==1.3.2
    pip install scikit-learn==1.4.0
    ```

3. Obtain a fully featured Trial License if you are from a private or public company, or Academic License if you are a student/professor at a university.

* Free licenses
  * To obtain a trial license go to <https://www.mosek.com/products/trial/>
  * To obtain a personal academic license go to <https://www.mosek.com/products/academic-licenses/>
  * To obtain an institutional academic license go to <https://www.mosek.com/products/academic-licenses/>
  * If you have a custom license go to <https://www.mosek.com/license/request/custom/> and enter the code you received.
* Commercial licenses
  * Assuming you purchased a product ( <https://www.mosek.com/sales/order/>) you will obtain a license file.

# 3.Neural Barrier Certificate Synthesis for NN-Controlled System 

Main steps as follows:

1. Add a new example you want to run at `/RL_train/Env.py`;
2. Modify `identity` in `benchmarks/run.py`;
3. Tuning hyperparameters and run it.

## 3.1 Case Study

You can create a new example at `RL_train/Env.py` as follows.

```python
1: Example(
            n_obs=2,  # the dimension of dynamic system.
            u_dim=1,  # the dimension of controller.
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),  # the location domain of system.
            I_zones=Zones('box', low=[-3, -3], up=[-1, -1]),  # the initial region of system.
            U_zones=Zones('box', low=[2, 1], up=[4, 3]),  # the unsafe region of system.
            f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] + u[0],  
               lambda x, u: -2 * x[1] - x[0] ** 2 + u[0]],  # differential equations of system.
            u=1,  # the output bound of controller. 
            path='C1/model',  # save path.
            dense=4,  #  the number of hidden layers for NN controller.
            units=20,  # the neuron's number of each hidden layer.
            activation='relu',  # the activation function. 
            id=1,  # identity.
            k=50 
        )
```
And you can get the NN controller and it's abstraction polynomial P(x), and then you can synthesize a neural barrier certificate by executing `benchmarks/run.py`.
```python
def main():

    identity = 1  # TODO: if you want to run example C{i}, let identity=i.
    trainer = RlTrain(identity, episodes=10, max_steps=1000)
    trainer.learn()

    fiter = PolFit(identity, iter=100, max_step=1000)
    p = fiter.fit()

    activations = ['MUL']  # 'MUL' denotes the 'quadratic' activation function .
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
        "MULTIPLICATOR_ACT": ["LINEAR"],
        # The activation function of each layer of the multiplier network;
        # since the last layer does not require an activation function, the number is one less than MULTIPLICATOR_NET.
        "BATCH_SIZE": 500,
        "LEARNING_RATE": 0.1,
        "MARGIN": 2.0,
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
```
[//]: # (## 3.2 Modify identity)

[//]: # ()
[//]: # (After you create the example you want, you can modify the parameter `identity` at `benchmarks/run.py` to the corresponding ID.)

[//]: # ()
[//]: # (## 3.3 Tuning hyperparameters and run it)

[//]: # ()
[//]: # (If you failed, adjust the hyperparameters and try again.)

## 3.2 Result

Finally, you can get the polynomial abstraction P(x) of NN controller and the barrier certificate B(x) for the system.

```python
P(x)=-0.01782780568647*x1**2 + 0.0304303727546807*x1*x2 + 0.0671740437924431*x1 - 0.234335039813611*x2**2 - 0.837997680548356*x2
B(x)=-0.164659815707088*x1**2 + 1.96691021817663*x1*x2 + 0.823110971728375*x1 - 2.31320037980075*x2**2 + 15.9294101325842*x2 - 3.10400062966616
```
Phase portrait of barrier certificate for the system in example C1:

![Barrier Certificate](https://github.com/blliu6/SNBC/tree/main/benchmarks/img/C1_2d.png)
![Barrier Certificate](https://github.com/blliu6/SNBC/tree/main/benchmarks/img/C1_3d.png)


