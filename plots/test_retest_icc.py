import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import DataFrame, FloatVector, IntVector

r_icc = importr("ICC")

path_session = ['/media/jrudascas/HDRUDAS/tesis/s1/output/datasink/preprocessing',
                '/media/jrudascas/HDRUDAS/tesis/s2/output/datasink/preprocessing']

path_parcellation = ['_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..AAL2.nii',
                     '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation.._parcellation_2mm.nii',
                     '_image_parcellation_path_..home..jrudascas..Desktop..Tesis..data..parcellations..rsn_parcellation..raw..Parcels_MNI_222.nii',
                     'parcellation_from_lasso']

measures = [#'correlation_matrix_base_univariate.txt',
            'correlation_matrix_distance_multivariate.txt']
            #'correlation_matrix_distance_multivariate_lagged.txt',
            #'correlation_matrix_distance_univariate.txt',
            #'correlation_matrix_distance_univariate_lagged.txt',
            #'correlation_matrix_pearson_univariate.txt',
            #'correlation_matrix_pearson_univariate_lagged.txt']

figure, axes = plt.subplots(nrows=1, ncols=len(measures), figsize=(10, 8))

cont = 0
measure_data_list = []
pos = [1,3,5]
for measure in measures:
    print(measure)
    icc_parcellation_list = []
    for parcellation in path_parcellation:
        print(parcellation)
        icc_session_list = []
        for id_session in range(len(path_session)):
            for id2_session in range(id_session + 1, len(path_session), 1):
                icc_list = []
                for subject in os.listdir(path_session[id_session]):

                    try:
                        connectivity_matrix_1 = np.loadtxt(join(path_session[id_session], subject, parcellation, measure),
                                                         delimiter=',').flatten()

                        connectivity_matrix_2 = np.loadtxt(join(path_session[id2_session], subject, parcellation, measure),
                                                         delimiter=',').flatten()

                        df = DataFrame({"x": FloatVector(connectivity_matrix_1),
                                        "y": FloatVector(connectivity_matrix_2)})

                        icc_res = r_icc.ICCbare(x="x", y="y", data=df)

                        icc_list.append(icc_res[0])
                    except:
                        pass
                icc_session_list.append(icc_list)
        t = np.asarray(icc_session_list)
        icc_parcellation_list.append(np.mean(t, axis=0))
    measure_data_list.append(icc_parcellation_list)

    axes[cont].violinplot(icc_parcellation_list, positions=pos, showmeans=True, showmedians=True)
    axes[cont].set_title(measure)
    cont += 1
plt.show()
