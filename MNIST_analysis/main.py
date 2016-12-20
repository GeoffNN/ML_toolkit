from MNIST_analysis.inout import read_MNIST_from_csv
from graphs.hard_hfs import hfs

X, Y = read_MNIST_from_csv()
labels = hfs(X, Y) # simple mode kills memory when establishing the graph
