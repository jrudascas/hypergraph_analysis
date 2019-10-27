from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class SignalExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    time_series_out_file = File(mandatory=True)
    correlation_matrix_out_file = File(mandatory=True)
    image_parcellation_path = traits.String(mandatory=True)
    labels_parcellation_path = traits.Either(
        traits.String(),
        traits.ArrayOrNone())
    mask_img = File(mandatory=True)
    tr = traits.Float(mandatory=True)
    low_pass = traits.Float(default_value=None)
    high_pass = traits.Float(default_value=None)
    plot = traits.Bool(default_value=False, mandatory=False)


class SignalExtractionOutputSpec(TraitedSpec):
    time_series_out_file = File(genfile=True)
    correlation_matrix_out_file = File(genfile=True)


class SignalExtraction(BaseInterface):
    input_spec = SignalExtractionInputSpec
    output_spec = SignalExtractionOutputSpec

    def _run_interface(self, runtime):

        from nilearn.input_data import NiftiLabelsMasker
        from nilearn.signal import clean
        import numpy as np
        import nibabel as nib

        '''
        masker = NiftiLabelsMasker(labels_img=self.inputs.image_parcellation_path,
                                   mask_img=self.inputs.mask_img,
                                   standardize=True,
                                   detrend=True,
                                   low_pass=self.inputs.low_pass,
                                   high_pass=self.inputs.high_pass,
                                   t_r=self.inputs.tr,
                                   memory='nilearn_cache',
                                   memory_level=5,
                                   verbose=0
                                   )
                                   
        time_series = masker.fit_transform(self.inputs.in_file)

        np.savetxt(self.inputs.time_series_out_file, time_series, fmt='%10.2f', delimiter=',')
        '''

        data_parcellation = np.squeeze(nib.load(self.inputs.image_parcellation_path).get_data())
        data_img = nib.load(self.inputs.in_file).get_data()

        time_series_list = []
        parcellation_index = np.unique(data_parcellation)

        for i in parcellation_index:
            time_series = data_img[data_parcellation == i, :]
            print(time_series.shape)

            ttt = [clean(t, standardize=True,
                         detrend=True,
                         low_pass=self.inputs.low_pass,
                         high_pass=self.inputs.high_pass,
                         t_r=self.inputs.tr) for t in time_series.tolist()]

            time_series_list.append(ttt)

            time_series.append(np.mean(data_fMRI[atlas_data == index, :], axis=0))

        labels = []
        if self.inputs.labels_parcellation_path is not None:
            file_labels = open(self.inputs.labels_parcellation_path, 'r')

            for line in file_labels.readlines():
                labels.append(line)
            file_labels.close()
        else:
            labels = list(range(time_series.shape[-1]))

        if self.inputs.plot:
            from nilearn import plotting
            from nilearn.connectome import ConnectivityMeasure
            import matplotlib
            import matplotlib.pyplot as plt
            fig, ax = matplotlib.pyplot.subplots()

            font = {'family': 'normal',
                    'size': 1}

            matplotlib.rc('font', **font)

            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform([time_series_list])[0]

            # Mask the main diagonal for visualization:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(correlation_matrix, figure=fig, labels=labels, vmax=0.8, vmin=-0.8, reorder=False)

            fig.savefig(self.inputs.correlation_matrix_out_file, dpi=1200)

        return runtime

    def _list_outputs(self):
        return {'time_series_out_file': os.path.abspath(self.inputs.time_series_out_file),
                'correlation_matrix_out_file': os.path.abspath(self.inputs.correlation_matrix_out_file)}

    def _gen_filename(self, name):
        if name == 'time_series_out_file':
            return os.path.abspath(self.inputs.time_series_out_file)
        if name == 'correlation_matrix_out_file':
            return os.path.abspath(self.inputs.correlation_matrix_out_file)
        return None
