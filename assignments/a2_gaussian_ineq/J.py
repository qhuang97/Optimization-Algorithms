"""
Sample code automatically generated on 2022-12-06 14:45:43

by www.matrixcalculus.org

from input

d/dx -exp(-(x-x0)'*D*(x-x0)) = 2*exp(-(x-x0)'*D*(x-x0))*D*(x-x0)

where

D is a symmetric matrix
x is a vector
x0 is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(D, x, x0):
    assert isinstance(D, np.ndarray)
    dim = D.shape
    assert len(dim) == 2
    D_rows = dim[0]
    D_cols = dim[1]
    assert isinstance(x, np.ndarray)
    dim = x.shape
    assert len(dim) == 1
    x_rows = dim[0]
    assert isinstance(x0, np.ndarray)
    dim = x0.shape
    assert len(dim) == 1
    x0_rows = dim[0]
    assert x_rows == x0_rows == D_cols == D_rows

    t_0 = (x - x0)
    t_1 = (D).dot(t_0)
    t_2 = np.exp(-(t_0).dot(t_1))
    functionValue = -t_2
    gradient = ((2 * t_2) * t_1)

    return functionValue, gradient

def checkGradient(D, x, x0):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3)
    f1, _ = fAndG(D, x + t * delta, x0)
    f2, _ = fAndG(D, x - t * delta, x0)
    f, g = fAndG(D, x, x0)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData():
    D = np.random.randn(3, 3)
    D = 0.5 * (D + D.T)  # make it symmetric
    x = np.random.randn(3)
    x0 = np.random.randn(3)

    return D, x, x0

if __name__ == '__main__':
    D, x, x0 = generateRandomData()
    functionValue, gradient = fAndG(D, x, x0)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(D, x, x0)
