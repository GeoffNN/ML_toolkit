from graphs.graph_init import build_graph, build_laplacian
import numpy as np


class OnlineCoverState:
    def __init__(self, centroids, nodes_to_centroids_map, centroids_to_nodes_map, R, taboo, max_n_centroids, graph_params, laplacian_params):
        self.centroids = centroids
        self.n_centroids = len(centroids)
        self.max_n_centroids = max_n_centroids
        self.nodes_to_centroids_map = nodes_to_centroids_map
        self.centroids_to_nodes_map = centroids_to_nodes_map
        self.graph_params = graph_params
        self.laplacian_params = laplacian_params
        self.R = R
        self.taboo = taboo

    def online_compute_solution(self, t, Y):
        n_classes = len(Y.unique())
        # Build graph on centroids
        W_tilda_q = build_graph(self.centroids, self.graph_params)
        v = np.zeros(self.n_centroids)
        for k in range(self.n_centroids):
            v[k] = (self.nodes_to_centroids_map == self.centroids_to_nodes_map[k]).sum()
        V = np.diag(v)
        W_q = V.dot(W_tilda_q).dot(V)
        L = build_laplacian(W_q, self.laplacian_params)

        Y_mapped = Y[self.centroids_to_nodes_map]

        l_idx = np.nonzero(Y_mapped)
        u_idx = np.nonzero(Y_mapped == 0)
        n_l = len(l_idx)

        y = -np.ones((n_l, n_classes))
        for i in range(n_l):
            y[i, Y_mapped[l_idx[i]]] = 1

        f = np.zeros((len(Y_mapped)), n_classes)
        f[l_idx] = y
        f[u_idx] = L[u_idx, u_idx].inv().dot(W_q[u_idx, l_idx]).dot(f[l_idx])

        new_label_confidence = f[self.centroids_to_nodes_map == t].max()
        new_label = np.argmax(f[self.centroids_to_nodes_map == t])
        new_label[abs(new_label_confidence) < 0.3] = 0
        return new_label

    def online_update_centroids(self, t, datum, num_labels):
        online_state = []
        return online_state
