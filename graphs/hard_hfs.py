import numpy as np


def hfs(X_l, Y_l, X_u, mode='simple'):
    if mode is 'simple':
        return simple_hfs(X_l, Y_l, X_u)
    elif mode is 'iterative':
        return iterative_hfs(X_l, Y_l, X_u)
    elif mode is 'online':
        return online_hfs(X_l, Y_l, X_u)


def simple_hfs(X_l, Y_l, X_u):
    pass


def iterative_hfs(X_l, Y_l, X_u):
    pass


def online_hfs(X_l, Y_l, X_u):
    pass
