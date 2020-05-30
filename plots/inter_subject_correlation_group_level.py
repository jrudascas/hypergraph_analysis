import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import plotly.express as px


def plot(group):
    path_session = '/media/jrudascas/HDRUDAS/tesis/' + group + '/output/datasink/preprocessing'

    path_parcellation = ['_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..AAL2.nii',
                         # '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation.._parcellation_2mm.nii',
                         '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation..raw..Parcels_MNI_222.nii',
                         'parcellation_from_lasso']

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
                for j in range(i + 1, len(subject_list), 1):
                    try:
                        iu1 = np.triu_indices(np.loadtxt(join(path_session, subject_list[0], parcellation, measure),
                                                         delimiter=',').shape[0], k=1)

                        connectivity_matrix_1_flatted = np.loadtxt(
                            join(path_session, subject_list[i], parcellation, measure), delimiter=',')[iu1].flatten()

                        connectivity_matrix_2_flatted = np.loadtxt(
                            join(path_session, subject_list[j], parcellation, measure), delimiter=',')[iu1].flatten()

                        similarity_value = pearsonr(connectivity_matrix_1_flatted, connectivity_matrix_2_flatted)

                        icc_session_list.append(
                            {'measure': measure_name.split('_unsigned')[0], 'parcellation': parcellation_name, 'comparation': str(i) + ' - ' + str(j), 'similarity': similarity_value[0]})
                    except Exception as e:
                        pass

    df = pd.DataFrame.from_dict(icc_session_list)
    fig = px.violin(df, y="similarity", x="measure", color="parcellation", box=True, points="all", hover_data=df.columns, title='Inter subject similarity - ' + group)
    fig.show()

group = 'controls'
plot(group)

group = 'mcs'
plot(group)

group = 'uws'
plot(group)