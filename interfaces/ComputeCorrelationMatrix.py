from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class ComputeCorrelationMatrixInputSpec(BaseInterfaceInputSpec):
    hypergraph_path = File(exists=True, mandatory=True)
    time_series_path = File(mandatory=True)
    correlation_matrix_out_file = File(mandatory=True)
    correlation_matrix_plot_out_file = File(mandatory=True)


class ComputeCorrelationMatrixOutputSpec(TraitedSpec):
    correlation_matrix_out_file = File(genfile=True)
    correlation_matrix_plot_out_file = File(genfile=True)


class ComputeCorrelationMatrix(BaseInterface):
    input_spec = ComputeCorrelationMatrixInputSpec
    output_spec = ComputeCorrelationMatrixOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        from nilearn import plotting
        import matplotlib.pyplot as plt
        import dcor
        from datetime import datetime

        hypergraph = np.loadtxt(self.inputs.hypergraph_path, delimiter=',')
        time_series = np.loadtxt(self.inputs.time_series_path, delimiter=',')

        hypergraph_shape = hypergraph.shape
        correlation_matrix = np.zeros(hypergraph_shape)

        threshold = 0.3

        hypergraph[np.where(hypergraph > threshold)] = 1
        hypergraph[np.where(hypergraph != 1)] = 0
        hypergraph = hypergraph.astype(bool)

        then = datetime.now()

        for i in range(hypergraph_shape[0]):
            print(i)
            for j in range(i + 1, hypergraph_shape[0], 1):
                correlation_matrix[i, j] = dcor.u_distance_correlation_sqr(time_series[:, hypergraph[i, :]],
                                                                           time_series[:, hypergraph[j, :]])
                #correlation_matrix[j, i] = correlation_matrix[i, j]

        figure = plt.figure(figsize=(6, 6))
        plotting.plot_matrix(correlation_matrix, figure=figure, vmax=1., vmin=0.)
        figure.savefig(self.inputs.correlation_matrix_plot_out_file, dpi=300)

        np.savetxt(self.inputs.correlation_matrix_out_file, correlation_matrix, delimiter=',', fmt='%10.2f')

        print('Total time: ', (datetime.now() - then).total_seconds())
        return runtime

    def _list_outputs(self):
        return {'correlation_matrix_plot_out_file': os.path.abspath(self.inputs.correlation_matrix_plot_out_file),
                'correlation_matrix_out_file': os.path.abspath(self.inputs.correlation_matrix_out_file)}

    def _gen_filename(self, name):
        if name == 'correlation_matrix_plot_out_file':
            return os.path.abspath(self.inputs.correlation_matrix_plot_out_file)
        if name == 'correlation_matrix_out_file':
            return os.path.abspath(self.inputs.correlation_matrix_out_file)
        return None