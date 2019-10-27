import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import nibabel as nib
import scipy.ndimage as ndim

parcellation_rsn_path = '/home/jrudascas/Desktop/Tesis/data/parcellations/Parcels_MNI_222.nii'
parcellation_rsn_labels_path = '/home/jrudascas/Desktop/Tesis/data/parcellations/Parcels.csv'
output_path = '/home/jrudascas/Desktop/'

n_groups = [1]

img = nib.load(parcellation_rsn_path)
data_img = img.get_data()
affine_img = img.affine

data = pd.read_csv(parcellation_rsn_labels_path)


network_list = data['Community'].unique().tolist()

indexs = np.unique(data_img)

data_img_copy = np.zeros(data_img.shape)

for i in indexs:
    if i != 0:
        row = data.loc[data['ParcelID'] == i]

        network_index = network_list.index(row['Community'].values[0])
        data_img_copy[np.where(data_img == i)] = network_index + 1

nib.save(nib.Nifti1Image(data_img_copy.astype(np.float32), affine_img), output_path + '_parcellation_2mm.nii')
