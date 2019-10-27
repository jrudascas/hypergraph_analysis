import numpy as np


def run(fmri_data, gm_data):
    time_courses = []
    for slide in range(1, gm_data.shape[-1] - 1, 3):
        for col in range(1, gm_data.shape[1] - 1, 3):
            for row in range(1, gm_data.shape[0] - 1, 3):

                gm_indexes = [gm_data[row - 1, col - 1, slide - 1],
                              gm_data[row - 1, col, slide - 1],
                              gm_data[row - 1, col + 1, slide - 1],
                              gm_data[row, col - 1, slide - 1],
                              gm_data[row, col, slide - 1],
                              gm_data[row, col + 1, slide - 1],
                              gm_data[row + 1, col - 1, slide - 1],
                              gm_data[row + 1, col, slide - 1],
                              gm_data[row + 1, col + 1, slide - 1],
                              gm_data[row - 1, col - 1, slide],
                              gm_data[row - 1, col, slide],
                              gm_data[row - 1, col + 1, slide],
                              gm_data[row, col - 1, slide],
                              gm_data[row, col, slide],
                              gm_data[row, col + 1, slide],
                              gm_data[row + 1, col - 1, slide],
                              gm_data[row + 1, col, slide],
                              gm_data[row + 1, col + 1, slide],
                              gm_data[row - 1, col - 1, slide + 1],
                              gm_data[row - 1, col, slide + 1],
                              gm_data[row - 1, col + 1, slide + 1],
                              gm_data[row, col - 1, slide + 1],
                              gm_data[row, col, slide + 1],
                              gm_data[row, col + 1, slide + 1],
                              gm_data[row + 1, col - 1, slide + 1],
                              gm_data[row + 1, col, slide + 1],
                              gm_data[row + 1, col + 1, slide + 1]]
                """

                indexGreyMatter = [greyMatterData[row, col, slide],
                                   greyMatterData[row, col + 1, slide],
                                   greyMatterData[row + 1, col, slide],
                                   greyMatterData[row + 1, col + 1, slide],
                                   greyMatterData[row, col, slide + 1],
                                   greyMatterData[row, col + 1, slide + 1],
                                   greyMatterData[row + 1, col, slide + 1],
                                   greyMatterData[row + 1, col + 1, slide + 1]]
                """
                if np.count_nonzero(gm_indexes) > len(gm_indexes) * 1 / 2:
                    """
                    indexfMRI = np.array([fMRIData[row, col, slide],
                                          fMRIData[row, col + 1, slide],
                                          fMRIData[row + 1, col, slide],
                                          fMRIData[row + 1, col + 1, slide],
                                          fMRIData[row, col, slide + 1],
                                          fMRIData[row, col + 1, slide + 1],
                                          fMRIData[row + 1, col, slide + 1],
                                          fMRIData[row + 1, col + 1, slide + 1]])
                    """
                    fmri_indexes = np.array([fmri_data[row - 1, col - 1, slide - 1],
                                             fmri_data[row - 1, col, slide - 1],
                                             fmri_data[row - 1, col + 1, slide - 1],
                                             fmri_data[row, col - 1, slide - 1],
                                             fmri_data[row, col, slide - 1],
                                             fmri_data[row, col + 1, slide - 1],
                                             fmri_data[row + 1, col - 1, slide - 1],
                                             fmri_data[row + 1, col, slide - 1],
                                             fmri_data[row + 1, col + 1, slide - 1],

                                             fmri_data[row - 1, col - 1, slide],
                                             fmri_data[row - 1, col, slide],
                                             fmri_data[row - 1, col + 1, slide],
                                             fmri_data[row, col - 1, slide],
                                             fmri_data[row, col, slide],
                                             fmri_data[row, col + 1, slide],
                                             fmri_data[row + 1, col - 1, slide],
                                             fmri_data[row + 1, col, slide],
                                             fmri_data[row + 1, col + 1, slide],

                                             fmri_data[row - 1, col - 1, slide + 1],
                                             fmri_data[row - 1, col, slide + 1],
                                             fmri_data[row - 1, col + 1, slide + 1],
                                             fmri_data[row, col - 1, slide + 1],
                                             fmri_data[row, col, slide + 1],
                                             fmri_data[row, col + 1, slide + 1],
                                             fmri_data[row + 1, col - 1, slide + 1],
                                             fmri_data[row + 1, col, slide + 1],
                                             fmri_data[row + 1, col + 1, slide + 1]])

                    time_courses.append(np.mean(fmri_indexes, axis=0))
    return time_courses
