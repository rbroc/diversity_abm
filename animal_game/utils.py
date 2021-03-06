from pathlib import Path
import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy, truncnorm
import math


def load_matrix(filename, index_col=0, path=None):
    ''' Load matrix from file
    Args:
        filename (str): name of the file to load
        index_col (str): name of the index column
        path (str): subfolder of current wd where to search 
    '''
    f_path = str(Path(path) / filename) if path else filename
    matrix = pd.read_csv(f_path, sep='\t', index_col=index_col)
    for i in range(matrix.shape[0]):
        matrix.iloc[i,i] = np.nan
    return matrix


def compute_thresholds(matrices, q=[.5, .6, .7, .8, .9], 
                       round_at=5, **kwargs):
    ''' Compute threshold values for specific quantiles 
    Args:
        matrix (np.ndarray): filenames similarity matrices
        q (int or iterable): list of quantiles to compute thresholds for '''
    for idx, mat in enumerate(matrices):
        m = load_matrix(mat, **kwargs)
        if idx == 0:
            a = np.nan_to_num(m.values.ravel())
        else: 
            a = np.append(a, np.nan_to_num(m.values.ravel()))
    v = [round(val, round_at) for val in np.quantile(a, q)]
    return dict(zip(q, v))


def compute_distance(m1, m2, method='norm', **kwargs):
    ''' Computes distance between matrices using numpy.linalg.norm,
        averaging, or computing number of non-matching values.
    Args:
        m1, m2 (np.ndarrays): input matrices
        method (str): which method to use to compute the distance.
            Must be one of 'norm', 'avg_dist', 'non_matching'
        kwargs: named arguments for numpy.linalg.norm
    '''
    m_diff = np.nan_to_num(m1) - np.nan_to_num(m2)
    if method == 'norm':
        dist = np.linalg.norm(m_diff, **kwargs)
    elif method == 'avg_dist':
        dist = np.nanmean(np.abs(m_diff.ravel()))
    elif method == 'non_matching':
        dist = len(np.where(m1 != m2)[0])
    elif method == 'avg_cross_entropy':
        dist = entropy(m1, m2, axis=0)
    else:
        raise ValueError("Method must be one of 'norm', 'avg_dist', "
                          "'non_matching'")
    return dist

def generate_turn_idx(nr_turns, agents):
    ''' Generate turn index '''
    a_list = agents * math.ceil(nr_turns / len(agents))
    a_list = a_list[:nr_turns]
    return list(zip(range(nr_turns), a_list))