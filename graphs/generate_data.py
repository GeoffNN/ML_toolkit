from sklean.datasets import make_blobs


def blobs(n_samples, dist_options):
    """Generates data from sklearn's Blob distribution"""
    n_blobs, blob_var = dist_options
    return make_blobs(n_samples, centers=n_blobs, cluster_std=blob_var)


def get_from_dataset(f=""):
    """Gets dataset from a tabular file"""
    if f is "":
        raise FileNotFoundError


