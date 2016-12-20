import numpy as np
from graphs.graph_init import build_graph, build_laplacian, GraphParams, LaplacianParams


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
    n_classes = len(np.unique(Y))

    # compute linear target for labelled samples
    l_idx = np.nonzero(Y)
    u_idx = np.nonzero(Y==0)
    n_l = len(l_idx)

    y = -np.ones((n_l, n_classes))
    for i in range(n_l):
        y[i, Y[l_idx[i]]] = 1

    # Compute solution
    f_l = y
    W = build_graph(X, graph_params)
    L = build_laplacian(W, laplacian_params)

    f_u = L[u_idx, u_idx].inv().dot(W[u_idx, l_idx]).dot(f_l)

    # Compute label assignment
    l_l = f_l.argmax()
    l_u = f_u.argmax()
    labels = np.zeros(n_samples)
    labels[l_idx] = l_l
    labels[u_idx] = l_u
    return labels


def iterative_hfs(X, Y, graph_params, laplacian_params):
    pass


def online_hfs(X, Y, graph_params, laplacian_params):
    pass
