from os import listdir, path
import shutil, os


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)

preprocessing_path = '/home/jrudascas/Desktop/Tesis/data/datasets/mcs/output/datasink/preprocessing/'
time_serie_relative_path = 'parcellation_from_lasso/time_series.txt'

new_folder_path = '/home/jrudascas/Desktop/to_transfer/'
makedir(new_folder_path)

for subject in sorted(listdir(preprocessing_path)):
    if subject != 'output':
        time_serie_full_path = path.join(preprocessing_path, subject, time_serie_relative_path)
        makedir(path.join(new_folder_path, subject))
        shutil.copy(time_serie_full_path, path.join(new_folder_path, subject, 'time_serie.txt'))
