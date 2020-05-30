import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import plotly.express as px

group_1 = 's1'
group_2 = 's2'
group_2 = 's3'

path_session = ['/media/jrudascas/HDRUDAS/tesis/' + group_1 + '/output/datasink/preprocessing',
                '/media/jrudascas/HDRUDAS/tesis/' + group_2 + '/output/datasink/preprocessing',
                '/media/jrudascas/HDRUDAS/tesis/' + group_2 + '/output/datasink/preprocessing']

path_parcellation = ['_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..AAL2.nii',
                     #'_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation.._parcellation_2mm.nii',
                     '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation..raw..Parcels_MNI_222.nii',
                     'parcellation_from_lasso']

parcellation_labels = ['AAL2', 'RSN', 'Func', 'Hyper']

measures = [
            #'correlation_matrix_base_univariate.txt',
            'correlation_matrix_pearson_univariate.txt',
            'correlation_matrix_pearson_univariate_lagged.txt',
            'correlation_matrix_distance_univariate.txt',
            'correlation_matrix_distance_univariate_lagged.txt',
            'correlation_matrix_distance_multivariate.txt',
            'correlation_matrix_distance_multivariate_lagged.txt'
            ]

plt.rcParams.update({'font.size': 12})


cont = 0
measure_data_list = []
pos = np.arange(1, 3*len(path_parcellation), 3)
icc_session_list = []


for measure in measures:
    measure_name = measure.split('correlation_matrix_')[1].split('.txt')[0]
    print(measure_name)
    icc_parcellation_list = []
    for parcellation in path_parcellation:
        parcellation_name = parcellation.split('..')[-1].split('.nii')[0]
        print('---> ' + parcellation_name)
        for subject in os.listdir(path_session[0]):
            icc_list = []
            for id_session in range(len(path_session)):
                for id2_session in range(id_session + 1, len(path_session), 1):

                    try:
                        iu1 = np.triu_indices(np.loadtxt(join(path_session[id_session], subject, parcellation, measure),
                                                         delimiter=',').shape[0], k=1)

                        connectivity_matrix_1_flatted = np.loadtxt(join(path_session[id_session], subject, parcellation, measure),
                                                         delimiter=',')[iu1].flatten()

                        connectivity_matrix_2_flatted  = np.loadtxt(join(path_session[id2_session], subject, parcellation, measure),
                                                         delimiter=',')[iu1].flatten()

                        similarity_value = pearsonr(connectivity_matrix_1_flatted , connectivity_matrix_2_flatted )

                        icc_list.append(similarity_value[0])
                    except Exception as e:
                        pass
            if len(icc_list) > 0:
                icc_session_list.append({'measure':measure_name, 'parcellation': parcellation_name, 'subject': subject, 'similarity': np.mean(icc_list)})

df = pd.DataFrame.from_dict(icc_session_list)
fig = px.violin(df, y="similarity", x="measure", color="parcellation", box=True, points="all", hover_data=df.columns, title='Intra subject similarity - Test-retest dataset')
fig.show()
