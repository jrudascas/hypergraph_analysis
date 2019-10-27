# Jorge Rudas - jrudascas@gmail.com
# August 2019

# This experiment want to check if using a resting state network analysis
# based on the average time series for each network produce
# diferent results respect to a hypergraph approach

from os.path import join, isdir
from os import listdir
import nibabel as nib
import numpy as np
import pandas as pd
from utils import *

preprocessed_data_path = ''
parcellation_rsn_labels_path = '/home/jrudascas/Desktop/Tesis/data/parcellations/Parcels-19cwpgu/Parcels.csv'
rsn_atlas_path = '/home/jrudascas/Desktop/Tesis/data/parcellations/Parcels-19cwpgu/Parcels_MNI_222.nii'
preprocessed_data_default_name = 'fmri_preprocessed.nii'

#######################################################################################################################
################################################## RSN based on ROI ###################################################
#######################################################################################################################

atlas = nib.load(rsn_atlas_path)
atlas_data = atlas.get_data()
data = pd.read_csv(parcellation_rsn_labels_path)
network_list = data['Community'].unique()

group_results = []
for group in sorted(listdir(preprocessed_data_path)):
    path_group = join(preprocessed_data_path, group)
    if isdir(path_group):
        similarity_subject = []
        for subject in sorted(listdir(path_group)):
            path_subject = join(path_group, subject)
            if isdir(path_subject):
                fc_sessions = []
                for session in sorted(listdir(path_subject)):
                    path_subject_session = join(path_subject, session)
                    if isdir(path_subject_session):
                        fmri_preprocessed = nib.load(join(path_subject_session, preprocessed_data_default_name))
                        fmri_preprocessed_data = fmri_preprocessed.get_data()
                        fmri_preprocessed_affine = fmri_preprocessed.affine

                        time_courses_raw = []
                        time_courses_average = []

                        #Extracting time courses
                        for network_name in network_list:
                            network = data.loc[data['Community'] == network_name]
                            network_centroids = network['ParcelID'].tolist()

                            tc = fmri_preprocessed_data[np.where(atlas_data in network_centroids), :]
                            time_courses_raw.append(tc)
                            time_courses_average.append(np.mean(tc, axis=0))

                        #Computing functional connectivity among time courses

                        fc_pearson = compute_functional_connectivity(time_courses_raw, metric='distance')
                        fc_distance = compute_functional_connectivity(time_courses_average, metric='pearson')

                        fc_sessions.append((fc_pearson, fc_distance))
                similarity_pearson = ks_test(fc_sessions[0][0], fc_sessions[1][0])
                similarity_distance = ks_test(fc_sessions[0][1], fc_sessions[1][1])

            similarity_subject.append((similarity_pearson, similarity_distance))
    group_results.append((group, similarity_subject))

#Generating plots

