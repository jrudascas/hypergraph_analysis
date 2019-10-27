from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class SmoothingInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    fwhm = traits.Float(default_value=None, mandatory=True)

class SmoothingOutputSpec(TraitedSpec):
    out_file = File(genfile=True)

class Smoothing(BaseInterface):
    input_spec = SmoothingInputSpec
    output_spec = SmoothingOutputSpec

    def _run_interface(self, runtime):

        from nilearn import image
        import nibabel as nib
        import numpy as np

        smoothed_img = image.smooth_img(self.inputs.in_file, self.inputs.fwhm)
        nib.save(nib.Nifti1Image(smoothed_img, nib.load(self.inputs.in_file).affine), 'smoothed.nii.gz')

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath('smoothed.nii.gz')}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath('smoothed.nii.gz')
        return None