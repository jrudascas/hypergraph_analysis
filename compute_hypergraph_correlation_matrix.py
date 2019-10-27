import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import dcor
from os import listdir, path
from os.path import join


def compute_faster_correlation_matrix_from_hypergraph(hypergraph, time_series, savefigure=None):
    hypergraph_shape = hypergraph.shape
    correlation_matrix = np.zeros(hypergraph_shape)
    hypergraph = hypergraph.astype(bool)

    for i in range(hypergraph_shape[0]):
        print(i)
        for j in range(i + 1, hypergraph_shape[0], 1):
            correlation_matrix[i, j] = dcor.u_distance_correlation_sqr(time_series[:, hypergraph[i, :]],
                                                                 time_series[:, hypergraph[j, :]])
            #correlation_matrix[j, i] = correlation_matrix[i, j]

    if savefigure is not None:
        figure = plt.figure(figsize=(6, 6))
        plotting.plot_matrix(correlation_matrix, figure=figure, vmax=1., vmin=0.)
        figure.savefig(savefigure, dpi=200)

    return correlation_matrix


preprocessing_path = '/media/jrudascas/HDRUDAS/tesis/controls/output/datasink/preprocessing'
hypergraph_path = '/home/jrudascas/Desktop/Tesis/data/datasets/test/sub-sub05676/parcellation_from_lasso/hypergraph_parcellation/_hypergraph_0.1.txt'
hypergraph = np.loadtxt(hypergraph_path, delimiter=',')

for subject in sorted(listdir(preprocessing_path)):
    output_path = join(preprocessing_path, subject, 'parcellation_from_lasso')
    time_series_path = join(output_path, 'time_series.txt')

    time_series = np.loadtxt(time_series_path, delimiter=',')

    correlation_matrix = compute_faster_correlation_matrix_from_hypergraph(hypergraph_path, time_series, savefigure='')
    np.savetxt(join(output_path, 'hypergraph_correlation_matrix.txt'), correlation_matrix, delimiter=',', fmt='%10.2f')
