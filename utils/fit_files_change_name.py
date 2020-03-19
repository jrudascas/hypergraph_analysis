from os import listdir, path
import shutil, os

preprocessing_path = '/media/jrudascas/HDRUDAS/hypergraph_data/s3/output/datasink/preprocessing'
o = '/media/jrudascas/HDRUDAS/tesis/s3/output/datasink/preprocessing'
for subject in sorted(listdir(preprocessing_path)):
    print(path.join(preprocessing_path, subject, 'hypergraph_correlation_lagged_matrix.txt'))
    print(path.join(o, subject.split('_subject_id_')[-1], 'parcellation_from_lasso', 'correlation_matrix_distance_multivariate_lagged.txt'))
    if os.path.exists(path.join(o, subject.split('_subject_id_')[-1], 'parcellation_from_lasso')):
        shutil.copy(path.join(preprocessing_path, subject, 'hypergraph_correlation_lagged_matrix.txt'), path.join(o, subject.split('_subject_id_')[-1], 'parcellation_from_lasso', 'correlation_matrix_distance_multivariate_lagged.txt'))
