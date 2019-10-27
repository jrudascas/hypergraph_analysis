from sklearn.linear_model import Lasso

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import numpy as np

def fun_broyden(x):
    f = (3 - x) * x + 1
    f[1:] -= x[:-1]
    f[:-1] -= 2 * x[1:]
    return f


def sparsity_broyden(n):
    sparsity = lil_matrix((n, n), dtype=int)
    i = np.arange(n)
    sparsity[i, i] = 1

    i = np.arange(1, n)
    sparsity[i, i - 1] = 1
    i = np.arange(n - 1)
    sparsity[i, i + 1] = 1
    return sparsity

n = 100000
x0_broyden = -np.ones(n)

res_3 = least_squares(fun_broyden, x0_broyden, jac_sparsity=sparsity_broyden(n))

print(res_3.cost)
print(res_3.optimality)