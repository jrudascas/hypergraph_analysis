from os import listdir, path
import shutil, os

path_from = '/home/jrudascas/Desktop/to_transfer/results/hypergraph_data/UWS/output/datasink/preprocessing'
path_to = '/media/jrudascas/HDRUDAS/tesis/UWS/output/datasink/preprocessing'

file_name_1 = 'hypergraph_correlation_matrix.txt'
file_name_2 = 'hypergraph_correlation_matrix_plot.png'

for subject_name in sorted(listdir(path_from)):
    print(subject_name)
    new_subject_name = subject_name.replace('_subject_id_', '')

    absolute_from_path = path.join(path_from, subject_name, file_name_1)
    absolute_to_path = path.join(path_to, new_subject_name, 'parcellation_from_lasso', file_name_1)
    shutil.copy(absolute_from_path, absolute_to_path)

    absolute_from_path = path.join(path_from, subject_name, file_name_2)
    absolute_to_path = path.join(path_to, new_subject_name, 'parcellation_from_lasso', file_name_2)
    shutil.copy(absolute_from_path, absolute_to_path)
