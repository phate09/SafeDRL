import numpy as np
import plotly.graph_objects as go
import sympy
import torch
from mpmath import iv
from sparsemax import Sparsemax
from sympy import Point3D, Plane, Symbol

from symbolic import unroll_methods


def vanilla1():
    sparsemax = Sparsemax(dim=-1)
    softmax = torch.nn.Softmax(dim=-1)

    logits = torch.randn(1, 2)
    logits.requires_grad = True
    print("\nLogits")
    print(logits)

    softmax_probs = softmax(logits)
    print("\nSoftmax probabilities")
    print(softmax_probs)

    sparsemax_probs = sparsemax(logits)
    print("\nSparsemax probabilities")
    print(sparsemax_probs)


def range1():
    size_x, size_y = 50, 50
    sparsemax = Sparsemax(dim=-1)
    softmax = torch.nn.Softmax(dim=-1)
    max_x, max_y = 10, 10
    min_x, min_y = -10, -10
    delta_x, delta_y = max_x - min_x, max_y - min_y
    x, y = np.linspace(min_x, max_x, size_x), np.linspace(min_y, max_y, size_y)
    ix = np.vstack([x for _ in range(x.shape[0])])
    perm = np.vstack(([ix], [ix.T])).T
    z1 = np.zeros((size_x, size_y))
    z2 = np.zeros((size_x, size_y))
    zsoft1 = np.zeros((size_x, size_y))
    zsoft2 = np.zeros((size_x, size_y))
    zlin1 = np.zeros((size_x, size_y))
    for i in range(size_x):
        for j in range(size_y):
            z1[i, j] = sparsemax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][0].numpy().item()
            z2[i, j] = sparsemax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][1].numpy().item()
            zsoft1[i, j] = softmax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][0].numpy().item()
            zsoft2[i, j] = softmax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][1].numpy().item()
            min_tensor = torch.tensor([min_x, min_y])
            zlin1[i, j] = ((torch.from_numpy(perm[i, j]) + min_tensor).sum() / (delta_x + delta_y)).item()
    fig1 = go.Figure(data=[go.Surface(z=z1, x=x, y=y), go.Surface(z=zsoft1, x=x, y=y), go.Surface(z=zlin1, x=x, y=y)])
    fig1.show()
    fig2 = go.Figure(data=[go.Surface(z=z2, x=x, y=y), go.Surface(z=zsoft2, x=x, y=y)])
    fig2.show()


def range2():
    '''Range with planes'''
    size_x, size_y = 50, 50
    sparsemax = Sparsemax(dim=-1)
    softmax = torch.nn.Softmax(dim=-1)
    max_x, max_y = 10, 10
    min_x, min_y = -10, -10
    delta_x, delta_y = max_x - min_x, max_y - min_y
    x, y = np.linspace(min_x, max_x, size_x), np.linspace(min_y, max_y, size_y)
    ix = np.vstack([x for _ in range(x.shape[0])])
    perm = np.vstack(([ix], [ix.T])).T
    z1 = np.zeros((size_x, size_y))
    z2 = np.zeros((size_x, size_y))
    zsoft1 = np.zeros((size_x, size_y))
    zsoft2 = np.zeros((size_x, size_y))
    zlin1 = np.zeros((size_x, size_y))
    a = Plane(Point3D(min_x, max_y, 1), Point3D(max_x, min_y, 0), Point3D(max_x, max_y, 0.5))
    x, y, z = [Symbol(i, real=True) for i in 'xyz']
    for i in range(size_x):
        for j in range(size_y):
            z1[i, j] = sparsemax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][0].numpy().item()
            z2[i, j] = sparsemax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][1].numpy().item()
            zsoft1[i, j] = softmax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][0].numpy().item()
            zsoft2[i, j] = softmax(torch.from_numpy(perm[i, j]).unsqueeze(0))[0][1].numpy().item()
            min_tensor = torch.tensor([min_x, min_y])
            zlin1[i, j] = np.float(sympy.solve(a.equation().subs({x: i, y: j}), z)[0])
    fig1 = go.Figure(data=[go.Surface(z=z1, x=x, y=y), go.Surface(z=zsoft1, x=x, y=y), go.Surface(z=zlin1, x=x, y=y)])
    fig1.show()  # fig2 = go.Figure(data=[go.Surface(z=z2, x=x, y=y), go.Surface(z=zsoft2, x=x, y=y)])  # fig2.show()


def softmax_interval_test1():
    a = iv.mpf([2, 3])
    b = iv.mpf([-2, 1])
    iv.exp(a) / (iv.exp(a) + iv.exp(b))  # softmax
    1 / (1 + iv.exp(-a))  # sigmoid
    x, y = sympy.symbols("x y")
    softmax_func = sympy.exp(x) / (sympy.exp(x) + sympy.exp(y))  # softmax of a, calculate interval by putting ub of a and lb of everything else. reverese for lb
    softmax_func2 = sympy.exp(y) / (sympy.exp(x) + sympy.exp(y))
    float(softmax_func.subs({x: 2, y: 1}))
    float(softmax_func.subs({x: 3, y: -2}))
    float(softmax_func2.subs({x: 2, y: 1}))
    float(softmax_func2.subs({x: 3, y: -2}))


def softmax_interval_test2():
    intervals = [(0, 2), (-1, 2), (2, 3)]
    results = unroll_methods.softmax_interval(intervals)
    print(results)





if __name__ == '__main__':
    range1()
