from scipy.spatial.distance import pdist

from graphs.graph_init import build_graph, build_laplacian, GraphParams, LaplacianParams
import numpy as np


class OnlineCoverState:
    def __init__(self, n_samples, max_n_centroids=20, graph_params=GraphParams(),
                 laplacian_params=LaplacianParams()):
        self.centroids = []
        self.n_centroids = 0
        self.max_n_centroids = max_n_centroids
        self.nodes_to_centroids_map = np.zeros(n_samples)
        self.centroids_to_nodes_map = np.zeros(max_n_centroids)
        self.graph_params = graph_params
        self.laplacian_params = laplacian_params
        self.R = 0

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
        if self.n_centroids == self.max_n_centroids:
            W = build_graph(self.centroids, self.graph_params)
            # Transform similarities into distances
            distances = 1. / (W + 10 ** (-30))

            # Put taboo and self loops to infinity
            distances[:, self.taboo] = np.inf
            distances[self.taboo, :] = np.inf
            np.fill_diagonal(distances, np.inf)

            idx = distances.argmax()
            c_add, c_rep = np.unravel_index(idx, distances.shape)
            dx = np.min(1. / (
                np.exp(-pdist(self.centroids, datum, 'euclidean') / (2 * self.graph_param.sigma2)) + 10 ** (-30)));

            if c_rep not in self.taboo and c_rep > num_labels:
                if np.sum(self.nodes_to_centroids_map == self.centroids_to_nodes_map[c_add]) > np.sum(
                                self.nodes_to_centroids_map == self.centroids_to_nodes_map[c_rep]):
                    auxv = c_rep
                    c_rep = c_add
                    c_add = auxv

            if self.R == 0:
                self.R = distances.min()
                # taboo is index: careful
                self.taboo = np.zeros(self.max_n_centroids, dtype=bool)
                self.taboo[num_labels] = True
                print("R = {}".format(self.R))
                print("{} quantization vertices".format(len(self.n_centroids)))
            elif dx > self.R:
                R *= 1.5
                self.taboo = np.zeros(self.max_n_centroids, dtype=bool)
                self.taboo[num_labels] = True
                print("R = {}".format(self.R))
                print("{} quantization vertices".format(len(self.n_centroids)))

            idx = np.nonzero(self.nodes_to_centroids_map == self.centroids_to_nodes_map[c_add])
            self.nodes_to_centroids_map[idx] = self.centroids_to_nodes_map[c_rep]

            self.taboo[c_rep] = True

            self.centroids_to_nodes[c_add] = t
            self.centroids += [datum]
            self.nodes_to_centroids_map[t] = c_add


        else:
            self.centroids_to_nodes_map.append(t)
            self.nodes_to_centroids_map[t] = t
            self.centroids.append[datum]
            self.taboo[t] = False
