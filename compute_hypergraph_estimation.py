from sklearn.linear_model import Lasso
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib
import dcor
from utils.utils import makedir
from os.path import join
import nibabel as nib
from nilearn.image import clean_img
from os import listdir, path
from utils.change_resolution import run as change_resolution

matplotlib.use('TkAgg')


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


def compute_correlation_matrix_from_hypergraph(hypergraph, time_series, delay=0, savefigure=None):
    correlation_matrix = np.zeros(hypergraph.shape)

    if delay == 0:
        k_circular = []
        k_circular.append(0)
    else:
        k_circular = range(-1 * delay, delay + 1, 1)

    hypergraph = hypergraph.astype(bool)

    for i in range(hypergraph.shape[0]):
        print(i)
        for j in range(i + 1, hypergraph.shape[0], 1):
            m = []
            for lag in k_circular:
                time_serie_lagged = np.roll(time_series[:, hypergraph[j, :]], lag)
                m.append(dcor.u_distance_correlation_sqr(time_series[:, hypergraph[i, :]], time_serie_lagged))

            correlation_matrix[i, j] = max(m)
            correlation_matrix[j, i] = correlation_matrix[i, j]

    if savefigure is not None:
        figure = plt.figure(figsize=(6, 6))
        plotting.plot_matrix(correlation_matrix, figure=figure, vmax=1., vmin=0.)
        figure.savefig(savefigure, dpi=200)

    return correlation_matrix


def compute_hypergraph(time_series, savefigure=None, alpha=0.1, threshold=0):
    # (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    print(alpha)
    clf = Lasso(alpha=alpha, max_iter=2000, tol=1e-3)

    hypergraph = np.zeros((time_series.shape[1], time_series.shape[1]))

    for i in range(time_series.shape[1]):
        X = time_series.copy()
        Y = X[:, i].copy()
        X[:, i] = 0

        hypergraph[i, :] = clf.fit(X, Y).coef_
        hypergraph[i, np.where(hypergraph[i, :] > threshold)] = 1
        hypergraph[i, np.where(hypergraph[i, :] < -threshold)] = 1
        hypergraph[i, np.where(hypergraph[i, :] != 1)] = 0

    if savefigure is not None:
        figure = plt.figure(figsize=(6, 6))
        plotting.plot_matrix(hypergraph, figure=figure, vmax=1., vmin=-1.)
        figure.savefig(savefigure, dpi=200)

    print('HyperGraph computing finished')
    return hypergraph


def run(ts_path, output_path, savefigure, faster=False):
    x = np.loadtxt(ts_path, delimiter=',')
    alphas = np.arange(0.1, 0.31, 0.05)

    output_path = join(output_path, 'hypergraph_parcellation')
    makedir(output_path)

    hypergraph_list = []
    for alpha in alphas:
        alpha = round(alpha, 2)
        print('Computing a HyperGraph with ' + str(alpha) + ' sparse level')
        hypergraph = compute_hypergraph(time_series=x, alpha=alpha, savefigure=savefigure + str(alpha) + '.png')
        hypergraph_list.append(hypergraph)
        np.savetxt(join(output_path, '_hypergraph_' + str(alpha) + '.txt'), hypergraph, delimiter=',', fmt='%i')

    #print(np.asarray(hypergraph_list).shape)
    #median_hypergraph = np.median(np.asarray(hypergraph_list), axis=0)

    #figure = plt.figure(figsize=(6, 6))
    #plotting.plot_matrix(median_hypergraph, figure=figure, reorder=False)
    #figure.savefig(join(output_path, '_hypergraph_median.png'), dpi=200)


    #if faster:
    #    print('Computing correlation matrix not delayed')
    #    correlation_matrix = compute_faster_correlation_matrix_from_hypergraph(hypergraph=median_hypergraph, time_series=x,
    #                                                                               savefigure=savefigure + str(
    #                                                                                   alpha) + '_correlation_matrix.png')
    #else:
    #    print('Computing correlation matrix delayed')
    #    correlation_matrix = compute_correlation_matrix_from_hypergraph(hypergraph=median_hypergraph, time_series=x,
    #                                                                        delay=0,
    #                                                                        savefigure=savefigure + str(
    #                                                                            alpha) + '_correlation_matrix.png')

        # print('Computing correlation matrix delayed')
        # correlation_matrix_lagged = compute_correlation_matrix_from_hypergraph(hypergraph=hypergraph, time_series=x,
        #                                                                       delay=3, savefigure=savefigure + str(
        #        alpha) + '_lagged.png')


        # np.savetxt(join(output_path, 'correlation_matrix_hypergraph_lagged_' + str(alpha) + '.txt'),
        #           correlation_matrix_lagged)

    return hypergraph_list

#preprocessing_path = '/home/jrudascas/Desktop/Tesis/data/datasets/test'
preprocessing_path = '/media/jrudascas/HDRUDAS/tesis/controls/output/datasink/preprocessing'
nmi_brain_mask_path = '/home/jrudascas/Desktop/Tesis/data/parcellations/MNI152_T1_2mm_brain_mask.nii.gz'
gm_mask_path = '/home/jrudascas/Desktop/Tesis/data/parcellations/GM_mask_MNI_2mm.nii'
threshold = 0.6

hypergraphs_list = []
for subject in sorted(listdir(preprocessing_path)):
    print(subject)
    fmri_preprocessed_path = join(preprocessing_path, subject, 'swfmri_art_removed.nii')
    gm_data = nib.load(gm_mask_path, mmap=False).get_fdata()
    gm_data_copy = gm_data.copy()
    where_are_NaNs = np.isnan(gm_data_copy)
    gm_data[where_are_NaNs] = 0
    gm_data[gm_data_copy >= threshold] = 1
    gm_data[gm_data_copy < threshold] = 0

    confunds_path = join(preprocessing_path, subject, 'ev_without_gs.csv')

    fmri_cleaned_path = join(preprocessing_path, subject, 'fmri_cleaned.nii')
    if not path.exists(fmri_cleaned_path):
        print('Cleaning image')
        image_cleaned = clean_img(fmri_preprocessed_path,
                                  sessions=None,
                                  detrend=True,
                                  standardize=True,
                                  low_pass=0.08,
                                  high_pass=0.009,
                                  t_r=2,
                                  confounds=confunds_path,
                                  ensure_finite=True,
                                  mask_img=nmi_brain_mask_path)
        nib.save(image_cleaned, fmri_cleaned_path)
    else:
        print('Image cleaned found')
        image_cleaned = nib.load(fmri_cleaned_path)

    folder_output = join(preprocessing_path, subject, 'parcellation_from_lasso')
    time_series_path = join(folder_output, 'time_series.txt')
    makedir(folder_output)

    if not path.exists(time_series_path):
        time_series = np.transpose(np.asarray(change_resolution(image_cleaned.get_data(), gm_data)))
        np.savetxt(time_series_path, time_series, delimiter=',', fmt='%10.2f')
        print('Time series Shape: ' + str(time_series.shape))

    hypergraphs = run(time_series_path, folder_output, savefigure=join(folder_output, 'hypergraph_'), faster=True)
    hypergraphs_list.append(hypergraphs)

print(np.asarray(hypergraphs_list).shape)
median_hypergraph = np.median(np.median(np.asarray(hypergraphs_list), axis=1), axis=0)

figure = plt.figure(figsize=(6, 6))
plotting.plot_matrix(median_hypergraph, figure=figure, reorder=False)
figure.savefig('_hypergraph_median.png', dpi=200)
np.savetxt(join('_hypergraph_median.txt'), median_hypergraph, delimiter=',', fmt='%10.1f',)

