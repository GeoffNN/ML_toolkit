import numpy as np
from scipy.spatial.distance import pdist

"""This code is based on Daniel Calandriello's code for the Graphs in ML course at MVA - ENS Cachan."""


class GraphParams:
    def __init__(self, type='knn', thresh=10, sigma2=1):
        self.type = type
        self.thresh = thresh
        self.sigma2 = sigma2


class GraphTypeError(Exception):
    pass


def similarities(X, sigma2, metric='euclidean'):
    sims = pdist(X, metric)
    return np.exp(-np.abs(sims) / sigma2)


def build_graph(X, graph_params=GraphParams(), metric='euclidean'):
    graph_type = graph_params.type
    graph_thresh = graph_params.thresh
    sigma2 = graph_params.sigma2

    sims = similarities(X, sigma2, metric)
    n = len(X)
    W = np.zeros((n, n))

    if graph_type is 'knn':
        i_idx = np.argsort(sims)[1:graph_thresh]
        D = sims[i_idx]
        j_idx = np.tile(range(n), (0, graph_thresh - 1))
        z = D[:, range(graph_thresh)]

        W[i_idx, j_idx] = z

    elif graph_type is 'eps':
        W = sims.where(sims >= graph_thresh)

    else:
        raise GraphTypeError("Not a valid graph type")

    return W
