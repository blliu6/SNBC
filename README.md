# 1.Introduction



* `/benchmarks`: the source code and some examples;
* `/learn`: the code of learners of barrier certificate;
* `/verify`: the code of verifiers;
* `/plots`: the code of plots;
* `/RL_learn`: the code of training controller by reinforcement learning;
* `/utils`: the configuration of the project.


# 2.Configuration

## 2.1 System requirements

To install and run SNBC, you need:

* Windows Platform: `Python 3.9`;
* Linux Platform: `Python 3.9`;
* Mac OS X Platform: `Python 3.9`.

## 2.2 Installation instruction

You need install required software packages listed below and setting up a MOSEK license .

1. Download SNBC.zip, and unpack it;
2. Install the required software packages for using SynNBC:

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

# 3.Generate a Controller and Fit it to a Polynomial

Main steps to generate a controller and fit it to a polynomial:

1. Add a new example you want to run at `/RL_learn/Env.py`;
2. Fix `identity` in `benchmarks/run.py`;
3. Tuning hyperparameters and run it.

## 3.1 New examples

You can create a new example at `RL_learn/Env.py` as follows.

```python
1: Example(
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('box', low=[-3, -3], up=[-1, -1]),
            U_zones=Zones('box', low=[2, 1], up=[4, 3]),
            f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] + u[0],
               lambda x, u: -2 * x[1] - x[0] ** 2 + u[0]],
            u=1,
            path='C1/model',
            dense=4,
            units=20,
            activation='relu',
            id=1,
            k=50
        )
```

## 3.2 Modify identity

After you create the example you want, you can modify the parameter `identity` at `benchmarks/run.py` to the corresponding ID.

## 3.3 Tuning hyperparameters and run it

If you failed, adjust the hyperparameters and try again.



