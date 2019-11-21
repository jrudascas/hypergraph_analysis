from os import listdir, path
from os.path import join
import numpy as np
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from os.path import join as opj
from nipype import Workflow, Node
import interfaces.ComputeCorrelationMatrix as c

preprocessing_path = '/home/jrudascas/Desktop/Tesis/data/datasets/mcs/output/datasink/preprocessing'
hypergraph_path = '/home/jrudascas/PycharmProjects/hypergraph_analysis/_hypergraph_median.txt'
correlation_matrix_out_file = 'hypergraph_correlation_matrix.txt'
correlation_matrix_plot_out_file = 'hypergraph_correlation_matrix_plot.png'

experiment_dir = join(preprocessing_path, 'output/')
output_dir = 'datasink'
working_dir = 'workingdir'

subject_list = ['sub-MCS_AI',
                'sub-MCS_BI',
                'sub-MCS_BM',
                'sub-MCS_CA',
                'sub-MCS_DE',
                'sub-MCS_DF',
                'sub-MCS_DM',
                'sub-MCS_DP',
                'sub-MCS_FI',
                'sub-MCS_FR',
                'sub-MCS_HE',
                'sub-MCS_HR',
                'sub-MCS_MF',
                'sub-MCS_MM',
                'sub-MCS_MO',
                'sub-MCS_MT',
                'sub-MCS_MU',
                'sub-MCS_NL',
                'sub-MCS_PA',
                'sub-MCS_PJ',
                'sub-MCS_RB',
                'sub-MCS_RP',
                'sub-MCS_SC',
                'sub-MCS_ST',
                'sub-MCS_TP',
                'sub-MCS_VW']

# Infosource-a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles-to grab the data (alternativ to DataGrabber)
time_serie_file = opj('{subject_id}', 'time_serie.txt')

templates = {'time_series': time_serie_file}
selectfiles = Node(SelectFiles(templates, base_directory=preprocessing_path), name="selectfiles")

datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir), name="datasink")

experiment_dir = opj(preprocessing_path, 'output/')
preproc = Workflow(name='preproc')
preproc.base_dir = opj(experiment_dir, working_dir)

compute_correlation_matrix = Node(
    c.ComputeCorrelationMatrix(hypergraph_path=hypergraph_path, correlation_matrix_out_file=correlation_matrix_out_file,
                             correlation_matrix_plot_out_file=correlation_matrix_plot_out_file),
    name='compute_correlation_matrix')

preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                 (selectfiles, compute_correlation_matrix, [('time_series', 'time_series_path')]),
                 (compute_correlation_matrix, datasink,
                  [('correlation_matrix_out_file', 'preprocessing.@correlation_matrix_out_file')]),
                 (compute_correlation_matrix, datasink,
                  [('correlation_matrix_plot_out_file', 'preprocessing.@correlation_matrix_plot_out_file')])])
plugin_args = {'n_procs':1}
preproc.run(plugin='MultiProc', plugin_args=plugin_args)
#14:33