import os
import numpy as np


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def ks_test(A, B):
    import scipy
    return 1 / scipy.stats.ks_2samp(np.ravel(A), np.ravel(B))[0]


def compute_functional_connectivity(time_courses, metric='pearson'):
    from dcor import u_distance_correlation_sqr
    from scipy.stats import pearsonr
    fc_matrix = np.zeros((len(time_courses), len(time_courses)))

    for i in range(len(time_courses)):
        for j in range(i, len(time_courses)):
            if metric == 'pearson':
                fc_matrix[i, j] = pearsonr(time_courses[i], time_courses[j])
            elif metric == 'distance':
                fc_matrix[i, j] = u_distance_correlation_sqr(time_courses[i], time_courses[j])

    return fc_matrix