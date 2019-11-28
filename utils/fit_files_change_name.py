from os import listdir, path
import shutil, os

preprocessing_path = '/media/jrudascas/HDRUDAS/tesis/MCS/output/datasink/preprocessing'

for subject in sorted(listdir(preprocessing_path)):
    shutil.move(path.join(preprocessing_path, subject, 'parcellation_from_lasso', 'hypergraph_correlation_matrix.txt'), path.join(preprocessing_path, subject, 'parcellation_from_lasso', 'correlation_matrix_distance_multivariate.txt'))
