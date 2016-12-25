import numpy as np
from numpy.linalg import inv

from graphs.generate_data import blobs
from graphs.graph_init import GraphParams, LaplacianParams, build_graph, build_laplacian


# TODO: refactor using fit/transform

def hfs(X, Y, graph_params=GraphParams(), laplacian_params=LaplacianParams(), mode='simple'):
    if mode is 'simple':
        return simple_hfs(X, Y, graph_params, laplacian_params)
    elif mode is 'iterative':
        return iterative_hfs(X, Y, graph_params, laplacian_params)
    elif mode is 'online':
        return online_hfs(X, Y, graph_params, laplacian_params)


def simple_hfs(X, Y, graph_params, laplacian_params):
    n_samples = len(X)
    n_classes = len(np.unique(Y)) - 1

    # compute linear target for labelled samples
    l_idx = np.nonzero(Y)[0]
    u_idx = np.nonzero(Y == 0)[0]
    n_l = len(l_idx)
    y = -np.ones((n_l, n_classes))
    for i in range(n_l):
        y[i, Y[l_idx[i]] - 1] = 1
    print(y)
    # Compute solution
    f_l = y
    W = build_graph(X, graph_params)
    L = build_laplacian(W, laplacian_params)
    f_u = inv(L[[[x] for x in u_idx], u_idx]).dot(W[[[x] for x in u_idx], l_idx]).dot(f_l)

    # Compute label assignment
    l_l = f_l.argmax(axis=1)
    l_u = f_u.argmax(axis=1)
    labels = np.zeros(n_samples)
    labels[l_idx] = l_l
    labels[u_idx] = l_u
    return labels + 1


def iterative_hfs(X, Y, graph_params, laplacian_params):
    pass


def online_hfs(X, Y, graph_params, laplacian_params):
    pass


X_t, Y_t = blobs(150, 3, 2)
X = np.append(X_t[:100], np.zeros((50, 2)), axis=0)
Y = np.append(Y_t[:100] + 1, np.zeros(50), axis=0)
labs = hfs(X, Y)
print((labs == (Y_t + 1)).mean())
