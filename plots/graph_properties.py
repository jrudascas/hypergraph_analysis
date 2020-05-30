import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
from os.path import join
from networkx.algorithms import approximation, centrality, assortativity, cluster, communicability_alg, efficiency_measures


def fast_abs_percentile(data, percentile=80):
    data = np.abs(data)
    data = data.ravel()
    index = int(data.size * .01 * percentile)
    # Partial sort: faster than sort
    data = np.partition(data, index)
    return data[index]


path_session = '/media/jrudascas/HDRUDAS/tesis/' + 's1' + '/output/datasink/preprocessing'

path_parcellation = ['parcellation_from_lasso',
                     '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..AAL2.nii',
                     # '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation.._parcellation_2mm.nii',
                     '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation..raw..Parcels_MNI_222.nii'
                     ]

measures = [
    'correlation_matrix_distance_multivariate.txt',
    'correlation_matrix_distance_multivariate_lagged.txt',
    'correlation_matrix_pearson_univariate_unsigned.txt',
    'correlation_matrix_pearson_univariate_lagged_unsigned.txt',
    'correlation_matrix_distance_univariate_unsigned.txt',
    'correlation_matrix_distance_univariate_lagged_unsigned.txt',
    'correlation_matrix_distance_multivariate_unsigned.txt',
    'correlation_matrix_distance_multivariate_lagged_unsigned.txt'
]

plt.rcParams.update({'font.size': 12})

subject_list = sorted(os.listdir(path_session))

icc_session_list = []

for measure in measures:
    measure_name = measure.split('correlation_matrix_')[1].split('.txt')[0]
    print(measure_name)
    for parcellation in path_parcellation:
        parcellation_name = parcellation.split('..')[-1].split('.nii')[0]
        print('---> ' + parcellation_name)

        for i in range(len(subject_list)):
            np_data = np.loadtxt(join(path_session, subject_list[0], parcellation, measure), delimiter=',')

            np_data = np.abs(np_data)

            percentile_threshold = fast_abs_percentile(np_data, percentile=30)
            diag_indices = np.diag_indices(np_data.shape[0])
            np_data[diag_indices] = 0
            np_data[np.where(np_data < percentile_threshold)] = 0

            G = nx.from_numpy_array(np_data)

            #Approximations and Heuristics
            #print('1')
            #k_components = approximation.k_components(G)
            #print('2')
            ###max_clique = approximation.max_clique(G)
            #print('3')
            ###large_clique_size = approximation.large_clique_size(G)
            #print('4')
            average_clustering = approximation.average_clustering(G)
            #print('5')
            min_weighted_dominating_set = approximation.min_weighted_dominating_set(G)
            #print('6')
            min_edge_dominating_set = approximation.min_edge_dominating_set(G)
            #print('7')
            min_maximal_matching = approximation.min_maximal_matching(G)
            #print('8')
            ramsey_R2 = approximation.ramsey_R2(G)
            #print('9')
            node_connectivity = approximation.node_connectivity(G)
            #print('10')
            metric_closure = approximation.metric_closure(G)
            #print('11')
            #steiner_tree = approximation.steiner_tree(G)
            #print('12')
            treewidth_min_degree = approximation.treewidth_min_degree(G)
            #print('13')
            min_weighted_vertex_cover = approximation.min_weighted_vertex_cover(G)

            # Centrality

            #print('14')
            degree_centrality = centrality.degree_centrality(G)
            #print('15')
            #in_degree_centrality = centrality.in_degree_centrality(G)
            #print('16')
            #out_degree_centrality = centrality.out_degree_centrality(G)
            #print('17')
            closeness_centrality = centrality.closeness_centrality(G)
            #print('18')
            harmonic_centrality = centrality.harmonic_centrality(G)
            #print('19')
            #percolation_centrality = centrality.percolation_centrality(G)

            # Assortativity

            #print('20')
            degree_assortativity_coefficient = assortativity.degree_assortativity_coefficient(G)
            #print('21')
            #numeric_assortativity_coefficient = assortativity.numeric_assortativity_coefficient(G)
            #print('22')
            degree_pearson_correlation_coefficient = assortativity.degree_pearson_correlation_coefficient(G)
            #print('23')
            average_neighbor_degree = assortativity.average_neighbor_degree(G)
            #print('24')
            average_degree_connectivity = assortativity.average_degree_connectivity(G)

            # Clustering
            #print('25')
            triangles = cluster.triangles(G)
            #print('26')
            transitivity = cluster.transitivity(G)
            #print('27')
            average_clustering = cluster.average_clustering(G)
            #print('28')
            square_clustering = cluster.square_clustering(G)
            #print('29')
            generalized_degree = cluster.generalized_degree(G)

            #Communicability
            #print('30')
            communicability = communicability_alg.communicability(G)

            #Efficiency
            #print('31')
            global_efficiency = efficiency_measures.global_efficiency(G)
            #print('32')
            #local_efficiency = efficiency_measures.local_efficiency(G)
            #print('fin')