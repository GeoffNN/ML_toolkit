from sklearn.datasets import make_blobs


def blobs(n_samples, n_blobs, blob_var):
    """Generates data from sklearn's Blob distribution"""
    return make_blobs(n_samples, centers=n_blobs, cluster_std=blob_var)


#TODO:
def get_from_dataset(f=""):
    """Gets dataset from a tabular file"""
    if f is "":
        raise FileNotFoundError

X, Y = blobs(100, 3, 2)