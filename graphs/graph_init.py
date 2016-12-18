import numpy as np
from scipy.spatial.distance import pdist

"""This code is based on Daniel Calandriello's code for Michal Valko's Graphs in ML course (2016) at MVA - ENS Cachan."""


class GraphParams:
    def __init__(self, type='knn', thresh=None, sigma2=1):
        self.type = type
        if thres is None:
            if type is 'knn':
                thresh = 10
            elif type is 'eps':
                thresh = 0
            else:
                raise GraphTypeError("Not a valid graph type")
        self.thresh = thresh
        self.sigma2 = sigma2


class GraphTypeError(Exception):
    pass


def similarities(X, sigma2, metric='euclidean'):
    sims = pdist(X, metric)
    return np.exp(-np.abs(sims) / sigma2)


def build_graph(X, graph_params=GraphParams(), metric='euclidean'):
    """Builds a graph (knn or epsilon) using defined similarity function"""
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

    return W
