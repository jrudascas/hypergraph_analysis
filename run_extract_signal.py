from interfaces.SignalExtraction import SignalExtraction
from interfaces.Smoothing import Smoothing
from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from os.path import join as opj
from nipype.interfaces.fsl import ExtractROI
import nipype.interfaces.spm as spm

image_parcellation_path = ['/home/jrudascas/Desktop/Tesis/data/parcellations/rsn_parcellation/raw/Parcels_MNI_222.nii']
                           #'/home/jrudascas/Desktop/Tesis/data/parcellations/AAL2.nii',
                           #'/home/jrudascas/Desktop/Tesis/data/parcellations/Parcels_MNI_222.nii']
labels_parcellation_path = '/home/jrudascas/Desktop/Tesis/data/parcellations/rsn_parcellation/raw/Parcels_minimally.csv'
input_path = '/home/jrudascas/Desktop/Tesis/data/hcp'
mcr_path = '/opt/mcr/v95'
spm_path = '/home/jrudascas/Desktop/Tesis/data/spm/spm12_r7487_Linux_R2018b/spm12/run_spm12.sh'
prefix = 'rfMRI_REST1_LR'
#prefix = 'rfMRI_REST1_RL'
#prefix = 'rfMRI_REST2_LR'
#prefix = 'rfMRI_REST2_RL'

#matlab_cmd = spm_path + ' ' + mcr_path + '/ script'
#spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
#print(matlab_cmd)
#print('SPM version: ' + str(spm.SPMCommand().version))


subject_list = ['103818']
                #'103818',
                #'105923',
                #'111312',
                #'114823',
                #'115320',
                #'122317',
                #'125525',
                #'130518',
                #'135528',
                #'137128',
                #'139839',
                #'144226',
                #'146129',
                #'149337',
                #'149741',
                #'151526',
                #'158035',
                #'169343',
                #'172332',
                #'175439',
                #'177746',
                #'185442',
                #'187547',
                #'192439',
                #'194140',
                #'195041',
                #'200109',
                #'200614',
                #'204521',
                #'250427',
                #'287248',
                #'433839',
                #'562345',
                #'599671',
                #'601127',
                #'627549',
                #'660951',
                #'662551',
                #'783462',
                #'859671',
                #'861456',
                #'877168',
                #'917255']


tr = 0.720
low_pass = 0.08
high_pass = 0.009
init_volume = 6
fwhm = 8
experiment_dir = opj(input_path, 'output/' + prefix + '/')
output_dir = 'datasink'
working_dir = 'workingdir'
templates = {'func': opj('{subject_id}', prefix + '/' + prefix + '_hp2000_clean.nii.gz'),
             'mask_img':  opj('{subject_id}', prefix + '/brainmask_fs.2.nii.gz')}

#smooth = Node(Smoothing(fwhm=fwhm), name='smoothing')
#smooth = Node(spm.Smooth(fwhm=fwhm), name="smooth") #SPM
signal_extraction = Node(SignalExtraction(time_series_out_file='time_series.csv',
                                          correlation_matrix_out_file='correlation_matrix.png',
                                          labels_parcellation_path=labels_parcellation_path,
                                          tr=tr,
                                          low_pass=low_pass,
                                          high_pass=high_pass,
                                          plot=True),
                         name='signal_extraction')

signal_extraction.iterables = [('image_parcellation_path', image_parcellation_path)]


datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir), name="datasink")
substitutions = [('_subject_id_', 'sub-')]

datasink.inputs.substitutions = substitutions

selectfiles = Node(SelectFiles(templates, base_directory=input_path), name="selectfiles")

#extract = Node(ExtractROI(t_min=init_volume, t_size=-1, output_type='NIFTI'), name="extract")  # FSL

infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = [('subject_id', subject_list)]

preproc = Workflow(name='preproc')
preproc.base_dir = opj(experiment_dir, working_dir)

preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                 (selectfiles, signal_extraction, [('func', 'in_file')]),
                 (selectfiles, signal_extraction, [('mask_img', 'mask_img')]),
                 (signal_extraction, datasink, [('time_series_out_file', 'preprocessing.@time_serie')]),
                 (signal_extraction, datasink, [('correlation_matrix_out_file', 'preprocessing.@correlation_matrix')])])

preproc.write_graph(graph2use='colored', format='png', simple_form=True)

# Run workflow
plugin_args = {'n_procs' : 1,
               'memory_gb' : 9}
preproc.run(plugin='MultiProc', plugin_args=plugin_args)