import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial.distance import squareform

"""This code is based on Daniel Calandriello's code for Michal Valko's Graphs in ML course (2016) at MVA - ENS Cachan."""


class GraphTypeError(Exception):
    pass


class GraphParams:
    def __init__(self, graph_type='knn', thresh=None, sigma2=1):
        self.type = graph_type
        self.thresh = thresh
        if thresh is None:
            if graph_type is 'knn':
                self.thresh = 10
            elif graph_type is 'eps':
                self.thresh = 0
        if graph_type not in ['knn', 'eps']:
            raise GraphTypeError("Not a valid graph graph_type")
        self.sigma2 = sigma2


def build_graph(X, graph_params=GraphParams(), metric='euclidean'):
    """Builds a graph (knn or epsilon) weight matrix W
    W is sparse - to be optimized somehow
    """
    graph_type = graph_params.type
    graph_thresh = graph_params.thresh
    sigma2 = graph_params.sigma2
    n = len(X)
    W = np.zeros((n, n))
    if graph_type is 'knn':
        D = kneighbors_graph(X, graph_thresh, metric=metric, mode='distance').toarray()
    elif graph_type is 'eps':
        D = radius_neighbors_graph(X, graph_thresh, metric=metric, mode='distance').toarray()
    W[D > 0] = np.exp(-D[D > 0] / sigma2)
    return W
