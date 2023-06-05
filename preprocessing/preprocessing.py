import glob
import os
from os.path import join
import nibabel as nib
from nipype.interfaces.dcmstack import MergeNifti
import nipype.interfaces.fsl as fsl
import numpy as np
import pandas as pd
from pathlib import Path
import typing
import contextlib
import xml.etree.ElementTree as ET
import subprocess
import shutil
import re
StrOrBytesPath = typing.Union[str, bytes, os.PathLike]


def FEAT(fsf_file: StrOrBytesPath,
         remove_reg_files: bool = False,
         force: bool = True) -> None:
    """
    Simple wrapper for FEAT tool from FSL.

    Parameters
    ----------
    fsf_file: StrOrBytesPath
        Path to a `.fsf` file
    remove_reg_files : bool, optional
        If set `True`, it removes some of the unnecessary files generated
        during the registration stage to save disk space.
    force : bool, optional
        If set False, it will try to look for the result and return them instead of
        redoing all computations.

    Returns
    ----------
    feat_dir: StrOrBytesPath
       Path of the output feat directory
    """
    # if the filtered_func_data.nii.gz file exists, we assume the previous FEAT analysis was ran correctly
    feat_dir = join(os.path.dirname(fsf_file), ".feat")
    filtered_func_file = join(feat_dir, "filtered_func_data.nii.gz")
    if not force and os.path.exists(filtered_func_file):
        return feat_dir

    # remove the .feat directory since existing .feat dirs will cause problems
    possible_feat_dirs = glob.glob(join(os.path.dirname(fsf_file), "*.feat"))
    if force and len(possible_feat_dirs):
        for possible_feat_dir in possible_feat_dirs:
            shutil.rmtree(possible_feat_dir)

    # suppress print statment used in FEAT()
    with contextlib.redirect_stdout(None):
        feat = fsl.FEAT()
        feat.inputs.fsf_file = fsf_file
        feat.terminal_output = "none"
        feat_output = feat.run()
        feat_dir = join(feat_output.outputs.feat_dir, ".feat")

    if remove_reg_files:
        reg_dir = join(feat_dir, "reg")
        reg_files = ["example_func.nii.gz", "example_func2highres.nii.gz",
                     "example_func2standard.nii.gz", "highres.nii.gz", "standard.nii.gz"]
        for reg_file in reg_files:
            file = join(reg_dir, reg_file)
            if os.path.exists(file):
                os.path.remove(file)

    return feat_dir


def create_masks_from_atlas(in_file: StrOrBytesPath,
                            roi_info: StrOrBytesPath = None,
                            out_dir: StrOrBytesPath = None,
                            force: bool = True):
    """
    Creates individual mask file for each ROI provided.
    WARNING: Only tested for AAL atlas.

    Parameters
    ----------
    in_file: StrOrBytesPath,
        Path to the atlas file. (A single *.nii or *.nii.gz file)
    roi_info: StrOrBytesPath
        XML file containing metadata for each ROI.
    out_dir: StrOrBytesPath
        Path to save the ROI masks in.
    force : bool
        if set `True` it regenerates all the masks. Else, it
        only generates the non-existant masks.

    Returns
    ----------
    mask_paths: np.ndarray[StrOrBytesPath]
        Array of masks paths.

    labels: np.ndarray[str]
        Array of masks labels.
    """
    atlas_file_name = Path(in_file).stem

    if roi_info is None:
        p, _ = os.path.splitext(in_file)
        roi_info = p + ".xml"

    if out_dir is None:
        out_dir = Path(in_file).parent
    elif not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ROIs = ET.parse(roi_info).getroot().find(
        "data").findall("label")
    labels = [ROI.find("name").text for ROI in ROIs]
    values = [int(ROI.find("index").text) for ROI in ROIs]
    img = nib.load(in_file)
    voxels = img.get_fdata()
    mask_paths = []
    for label, value in zip(labels, values):
        mask_path = join(out_dir, f"{atlas_file_name}_{label}.nii.gz")
        if not os.path.exists(mask_path) or force:
            roi_voxels = np.where(voxels == value, 1, 0)
            out = nib.Nifti1Image(
                roi_voxels, header=img.header, affine=img.affine)
            nib.save(out, mask_path)
        mask_paths.append(mask_path)

    return np.array(mask_paths), np.array(labels)


def apply_threshold(in_file: StrOrBytesPath,
                    threshold: float,
                    out_file: StrOrBytesPath = None) -> None:
    """
    Creates a binary mask based on the value of the threshold

    Parameters
    ----------
    in_file: StrOrBytesPath
        Path to the input file. (A single *.nii or *.nii.gz file)
    threshold: float
        Threshold to apply to the image. Each element larger or equal to
        `threshold` is set to 1 and else are set to 0
    out_file: StrOrBytesPath
        Path to the output file. (A single *.nii or *.nii.gz file)
    """
    if out_file is None:
        out_file = in_file

    img = nib.load(in_file)
    voxels = img.get_fdata()

    new_voxels = np.where(voxels >= threshold, 1, 0)
    out = nib.Nifti1Image(
        new_voxels, header=img.header, affine=img.affine)
    nib.save(out, out_file)


def brain_extraction(in_file: StrOrBytesPath,
                     out_file: StrOrBytesPath = None,
                     frac: float = 0.5,
                     robust: bool = True,
                     reduce_bias: bool = False,
                     force: bool = True):
    """
    Simple wrapper for BET(Brain Extraction Tool) for FSL.

    Parameters
    ----------
    in_file: StrOrBytesPath
        Path to the input file. (A single *.nii or *.nii.gz file)
    out_file: StrOrBytesPath, optional
        Path to the output file. (A single *.nii or *.nii.gz file)
        If not provided, the file would be saved with same name and '_brain' added to the end.
    force : bool, optional
        If set False, it will try to look for the result and return them instead of
        redoing all computations.

    Returns
    ----------
    out_file: StrOrBytesPath, optional
        Path to the output file. (A single *.nii.gz file)

    """
    in_filename = Path(in_file).stem
    in_parent_path = Path(in_file).parent

    if out_file is None:
        out_file = join(in_parent_path, in_filename + "_brain.nii.gz")

    if not force and os.path.exists(out_file):
        return out_file

    bet = fsl.BET()
    bet.inputs.in_file = in_file
    if robust and reduce_bias:
        bet.inputs.reduce_bias = reduce_bias
    elif robust:
        bet.inputs.robust = robust
    elif reduce_bias:
        bet.inputs.reduce_bias = reduce_bias

    bet.inputs.frac = frac
    bet.inputs.vertical_gradient = 0
    bet.inputs.out_file = out_file
    bet.run()

    return out_file


def robust_brain_extraction(in_file: StrOrBytesPath,
                            out_file: StrOrBytesPath = None,
                            initial_frac: float = 0.5,
                            reduce_bias: bool = False,
                            robust: bool = True,
                            force: bool = True):
    """A more robust brain extraction where BET is ran multiple times
    until an acceptable result is achieved.
    Running BET with the same parameteres for an entire dataset might not
    give good results. So a new process is proposed to find the suitable parameters
    by running BET multiple times

    Process steps:
    1-  Normal BET with the provided parameteres is ran first

    2-  Check the values of the voxels in the neck area. If there are too many non-zero voxels
        run BET with the `reduce_bias` flag set as `True`

    3-  Compare the ratio of brain voxels to the total number of voxels to `acceptable_voxel_ratio_range`
        For example, if the ratio is lower than the range it means the previous BET missed
        some parts of the brain. So the fractional intensity threshold is lowered to cover
        more of the brain. And vice versa.

    Parameters
    ----------
    in_file : StrOrBytesPath
        _description_
    out_file : StrOrBytesPath, optional
        _description_, by default None
    initial_frac : float, optional
        _description_, by default 0.5
    reduce_bias : bool, optional
        _description_, by default False
    robust : bool, optional
        _description_, by default True
    force : bool, optional
        _description_, by default True
    """
    raise NotImplementedError()
    pass


def apply_matrix_to_mask(mask: StrOrBytesPath,
                         label: str,
                         matrix: StrOrBytesPath,
                         ref: StrOrBytesPath,
                         out_dir: StrOrBytesPath,
                         force: bool = True):
    """
    Applys a 4*4 transform matrix to a mask

    Parameters
    ----------
    mask: StrOrBytesPath
        Path to the mask file. (A single *.nii or *.nii.gz file)
    label: str
        Label of the mask
    matrix: StrOrBytesPath
        A file with FSL's default `.mat` format for transform matrices
    ref StrOrBytesPath
        Path to the refrence file. (A single *.nii or *.nii.gz file)
    out_dir: StrOrBytesPath
        Path to an existing directory for saving the transformed mask.
    force : bool, optional
        If set False, it will try to look for the result and return them instead of
        redoing all computations.

    Returns
    ----------
    out_file: StrOrBytesPath, optional
        Path to the output file. (A single *.nii or *.nii.gz file)
    """
    out_file = join(out_dir, f"roi_mask_{label}.nii.gz")
    out_file = apply_affine_transform(mask, matrix, ref, out_file, force)
    apply_threshold(out_file, 0.5)
    return out_file


def apply_affine_transform(in_file: StrOrBytesPath,
                           matrix: StrOrBytesPath,
                           ref: StrOrBytesPath,
                           out_file: StrOrBytesPath,
                           force: bool = True):
    """Simple wrapper for `flirt` tool in FSL with the `applyXFM` flag

    Parameters
    ----------
    in_file : StrOrBytesPath
        Path to the input file. (A single *.nii or *.nii.gz file)
    matrix : StrOrBytesPath
        A file with FSL's default `.mat` format containing the affine transformation matrix
    ref : StrOrBytesPath
        Path to the refrence file. (A single *.nii or *.nii.gz file)
    out_file : StrOrBytesPath
        Path to the output file. (A single *.nii or *.nii.gz file)
    force : bool, optional
        If set False, it will try to look for the result and return them instead of
        redoing all computations, by default True

    Returns
    -------
    out_file: StrOrBytesPath, optional
        Path to the output file. (A single *.nii or *.nii.gz file)
    """
    if not force and os.path.exists(out_file):
        return out_file

    applyxfm = fsl.preprocess.ApplyXFM()

    applyxfm.inputs.in_file = in_file
    applyxfm.inputs.in_matrix_file = matrix
    applyxfm.inputs.reference = ref

    applyxfm.inputs.out_file = out_file
    applyxfm.inputs.out_matrix_file = re.sub(r"\.nii(\.gz)?", ".mat", out_file)
    applyxfm.inputs.apply_xfm = True

    applyxfm.terminal_output = "none"
    applyxfm.run()
    return out_file


def get_mean_timeseries(in_file: StrOrBytesPath,
                        mask: StrOrBytesPath,
                        out_file: StrOrBytesPath = None,
                        force: bool = True):
    """
    Simple wrapper for `fslmeants` tool in FSL.
    Extracts average voxel intensity in the area covered by the provided mask
    in each volume.

    Parameters
    ----------
    in_file: StrOrBytesPath
        Path to the input file. (A single *.nii or *.nii.gz file)
    mask:  StrOrBytesPath
        Path to the mask.  (A single *.nii or *.nii.gz file)
    out_file: StrOrBytesPath, optional
        Path to the output file. (A single *.nii or *.nii.gz file)
        If no output directory is specified, a temporary file would be saved in /tmp directory
        and deleted immediately.
    force : bool, optional
        If set False, it will try to look for the result and return them instead of
        redoing all computations.
    Returns
    ----------
    timeseries: np.ndarray
        Array of average voxel intensity in each volume.
    """
    delete = False
    if out_file is None:
        out_file = "/tmp/fsl_timeseries_tmp.txt"
        delete = True
    if not force and os.path.exists(out_file):
        return out_file

    fslmeants = fsl.ImageMeants()
    fslmeants.inputs.in_file = in_file
    fslmeants.inputs.mask = mask
    fslmeants.inputs.out_file = out_file

    fslmeants.terminal_output = "none"
    fslmeants.run()
    with open(out_file, "r") as f:
        timeseries = f.read()
    timeseries = np.array(timeseries.split(), dtype=float)
    if delete:
        os.remove(out_file)

    return timeseries


def get_mean_timeseries_py(in_files: typing.List[StrOrBytesPath],
                           atlas: StrOrBytesPath,
                           roi_info: StrOrBytesPath = None) -> pd.DataFrame:
    """
    Alternative way of extracting timeseries data from fMRI
    without using any FSL tool.

    Parameters
    ----------
    in_files : list[StrOrBytesPath]
        A list of paths to the fMRI volumes. (List of *.nii or *.nii.gz files)
    atlas : StrOrBytesPath,
        Path to the atlas file. (A single *.nii or *.nii.gz file)
    roi_info: StrOrBytesPath
        XML file containing metadata for each ROI, by default None

    Returns
    -------
    pd.DatafFrame
        A pandas DataFrame of ROIs' timeseries data
    """

    if roi_info is None:
        p, _ = os.path.splitext(atlas)
        xml_file = p + ".xml"

    ROIs = ET.parse(xml_file).getroot().find(
        "data").findall("label")
    ROI_labels = [ROI.find("name").text for ROI in ROIs]
    ROI_values = [int(ROI.find("index").text) for ROI in ROIs]

    timeseries = pd.DataFrame(
        np.zeros((len(in_files), len(ROI_labels))), columns=ROI_labels)

    atlas_voxels = nib.load(atlas).get_fdata()

    for i, in_file in enumerate(in_files):
        in_file_voxels = nib.load(in_file).get_fdata()
        for ROI_value, ROI_label in zip(ROI_values, ROI_labels):
            ROI = in_file_voxels[atlas_voxels == ROI_value]
            if len(ROI) > 0:
                timeseries[ROI_label][i] = ROI.mean()
            else:
                timeseries[ROI_label][i] = 0
    return timeseries


def apply_mask(in_file: StrOrBytesPath,
               mask: StrOrBytesPath,
               out_file: StrOrBytesPath):
    apply_mask = fsl.ApplyMask()
    apply_mask.inputs.in_file = in_file
    apply_mask.inputs.mask_file = mask
    apply_mask.inputs.out_file = out_file
    apply_mask.terminal_output = "none"
    apply_mask.run()


def merge_nifti(in_dir: StrOrBytesPath,
                TR: float,
                out_dir: StrOrBytesPath = None,
                force: bool = True):
    """
    Simple wrapper for `fslmerge` tool in FSL
    * If there are a lot of files and/or they have long file or path names,
    the generated command line could hit linux' shell command character limit.
    In that case, it's advised to use `merge_nifti2` instead.

    Parameters
    ----------
    in_dir : StrOrBytesPath
        Directory containing `.nii` files to be merged. Files will be merged
        in alphabetical order.
    TR : float
        Value of Repitition time(TR) in seconds
    out_dir : StrOrBytesPath, optional
        Path to the save directory, by default None
    force : bool, optional
        If set False, it will try to look for the result and return them instead of
        redoing all computations, by default True

    Returns
    -------
    StrOrBytesPath
        Path to the merged file
    """
    files = glob.glob(in_dir + "/*.nii")

    if out_dir is None:
        out_file = os.path.splitext(files[0])[
            0] + "_merged.nii.gz"
    else:
        out_file = os.path.join(
            out_dir, Path(files[0]).stem + "_merged.nii.gz")

    if not force and os.path.exists(out_file):
        return out_file

    merger = fsl.Merge()
    merger.inputs.in_files = in_dir + "/*.nii"
    merger.inputs.merged_file = out_file
    merger.inputs.dimension = 't'
    merger.inputs.output_type = 'NIFTI_GZ'
    merger.inputs.tr = TR
    merger_output = merger.run()
    return merger_output.outputs.merged_file


def merge_nifti2(in_dir: StrOrBytesPath,
                 TR: float,
                 out_dir: StrOrBytesPath = None,
                 force: bool = True):
    """
    Same behavior as `merge_nifti` but long file names will not cause problem.

    Parameters
    ----------
    in_dir : StrOrBytesPath
        Directory containing `.nii` files to be merged. Files will be merged
        in alphabetical order.
    TR : float
        Value of Repitition time(TR) in seconds
    out_dir : StrOrBytesPath, optional
        Path to the save directory, by default None
    force : bool, optional
        If set False, it will try to look for the result and return them instead of
        redoing all computations, by default True

    Returns
    -------
    StrOrBytesPath
        Path to the merged file
    """

    files = glob.glob(in_dir + "/*.nii")

    if out_dir is None:
        out_file = os.path.splitext(files[0])[
            0] + "_merged.nii.gz"
    else:
        out_file = os.path.join(
            out_dir, Path(files[0]).stem + "_merged.nii.gz")

    command = f"fslmerge -tr {out_file} {in_dir}/*.nii {TR}"

    if not force and os.path.exists(out_file):
        return out_file
    subprocess.run(command, shell=True, check=True)
    return out_file


def split_fmri(in_file: StrOrBytesPath,
               out_base_name: StrOrBytesPath):
    """Simple wrapper for `fslsplit` tool in FSL

    Parameters
    ----------
    in_file : StrOrBytesPath
        Path to the input file. (A single *.nii or *.nii.gz file)
    out_base_name : StrOrBytesPath
        Output files prefix

    Returns
    -------
    list[StrOrBytesPath]
        A list of paths to the splitted fMRI volumes. (List of *.nii or *.nii.gz files)
    """
    n_vols = nib.load(in_file).shape[-1]
    in_dir = os.path.dirname(in_file)
    out_dirs = [join(in_dir, out_base_name +
                     f"{vol:04}.nii.gz") for vol in range(n_vols)]

    if all([os.path.isfile(out_dir) for out_dir in out_dirs]):
        return out_dirs

    splitter = fsl.Split()
    splitter.inputs.in_file = in_file
    splitter.inputs.dimension = 't'
    splitter.inputs.out_base_name = out_base_name

    splitter.run()
    return out_dirs
