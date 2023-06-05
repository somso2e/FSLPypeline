import os
from os.path import join
from pathlib import Path
import shutil
from time import time
from tqdm import tqdm
import multiprocessing as mp
import preprocessing as mrp
from Utils import create_logger
import pandas as pd
import glob
from datetime import datetime

"""
If set True, all process will redo all processes.
If set False, it will only do the processes that were not successful or haven't been done yet.
"""
FORCE_REGENERATE = False
"""
Arbitrary limit for fMRI volume counts. Depending on your hardware,
scans with too many volumes might cause a crash due to low memory
"""
MAX_FMRI_VOLUME_COUNT = 300
MIN_FMRI_VOLUME_COUNT = 50

''' There are 3 ways of extracting timeseries data from fMRI
    1-  Registering the preprocessed fMRI to standard space then extracting timeseries
        data based on the atlas masks using FSLmeants
        * Registering fMRI to standard space as a whole might require large amounts of RAM

    2-  Registering the atlas masks to the fMRI space then extracting timeseries data
        based on the registered masks using FSLmeants
        * This method might not be as accurate since when a affine transform is applied
        to the masks, they lose their binary status so there needs to be a threshold applied
        to them to turn them back to 0s and 1s. Here we used 0.5 threshold.

    3-  Splitting fMRI in to volumes then Registering each seperately in to the standard
        space. Then extracting timeseries data from each volume using python and numpy
        methods and concatenating them at the end.
        * Functionally the same as method 1 but without the need of large RAM

    *   Both method 1 and 3 require more disk space. It's advised to delete
        the transferred files immediately after to save disk space
'''
TIMESERIES_EXTRACTION_METHOD = 3
REMOVE_TRANSFERRED_FMRI_FILES = True

# Specify paths to input and output flies
DATASET_DIR = "/box/ADNI/"
ATLAS = "/box/AAL/ROI_MNI_V5.nii"
STANDARDS = "/root/fsl/data/standard"
MNI_2mm = join(STANDARDS, "MNI152_T1_2mm_brain.nii.gz")
REPETITION_TIME_TABLE_CSV_PATH = "/box/fmri_scans_repetitiontimes_TR.csv"

SAVE_DIR = "/box/outputs"
MASKS_SAVE_DIR = join(SAVE_DIR, "masks")
PREPROCESSED_OUT_DIR = join(SAVE_DIR, "preprocessed")
TIMESERIES_SAVE_DIR = join(SAVE_DIR, "timeseries")
time_of_start_fstr = datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
logs_path = join(SAVE_DIR, f"og_{time_of_start_fstr}.log")

Path(MASKS_SAVE_DIR).mkdir(exist_ok=True, parents=True)
Path(TIMESERIES_SAVE_DIR).mkdir(exist_ok=True, parents=True)

ATLAS_MASKS, ATLAS_LABELS = mrp.create_masks_from_atlas(ATLAS,
                                                        out_dir=MASKS_SAVE_DIR,
                                                        force=FORCE_REGENERATE)


tr_df = pd.read_csv(REPETITION_TIME_TABLE_CSV_PATH)


def main(indir):
    logger = create_logger(logs_path)
    t_start = time()
    indir = Path(indir)
    session = indir.parts[-1]
    ID = indir.parts[-2]

    progress = {
        "ID": ID,
        "session": session,
        "init": False,
        "nifti merge": False,
        "BET": False,
        "FEAT": False,
        "timeseries": False,
        "finish": False,
        "invalid": False,
    }
    session_outdir = join(PREPROCESSED_OUT_DIR, ID, session)

    logger.info(f"{ID}/{session}:STARTED...")

    # Create a 'func' and 'anat' folder for each session for saving intermidiete files
    func_outdir = join(session_outdir, "func")
    Path(func_outdir).mkdir(parents=True, exist_ok=True)
    anat_outdir = join(session_outdir, "anat")
    Path(anat_outdir).mkdir(parents=True, exist_ok=True)

    func_indir = join(indir, "func")
    anat_indir = join(indir, "anat")

    progress["init"] = True
    ##################################
    TR = float(tr_df.loc[(tr_df["Subject ID"] == ID) &
                         (tr_df["VISCODE"] == session)]["REPETITIONTIME"])

    # Discard sessions with too high and too low volume counts
    fmri_volume_count = len(glob.glob(join(func_indir, "*.nii")))
    if fmri_volume_count > MAX_FMRI_VOLUME_COUNT or fmri_volume_count < MIN_FMRI_VOLUME_COUNT:
        logger.error(
            f"{ID}/{session}:SCAN FLAGGED AS INVALID FOR HAVING TOO MANY OR TOO FEW VOLUMES. N={fmri_volume_count}")
        progress["invalid"] = True
        return progress

    # Merge `.niftii` functional files
    logger.info(f"{ID}/{session}:NIFTI MERGE...")
    try:
        func_image = mrp.merge_nifti2(func_indir, TR,
                                      func_outdir, force=FORCE_REGENERATE)
    except Exception as e:
        logger.error(f"{ID}/{session}:ERROR IN NIFTI MERGE:\n" + str(e))
        return progress
    progress["nifti merge"] = True
    ##################################
    logger.info(f"{ID}/{session}:BET...")

    anat_image = (glob.glob(anat_indir + "/*.nii") +
                  glob.glob(anat_indir + "/*.nii.gz"))[0]
    try:
        # Structural image brain Extraction
        smri_BET = mrp.brain_extraction(anat_image,
                                        join(anat_outdir,
                                             f"T1_BET_{ID}_brain.nii.gz"),
                                        force=FORCE_REGENERATE)
    except Exception as e:
        logger.error(f"{ID}/{session}:ERROR IN BET:\n" + str(e))
        return progress
    progress["BET"] = True
    ##################################
    # Generate .fsf design file for preprocesssing fMRI
    logger.info(f"{ID}/{session}:FEAT...")
    try:
        slice_timing_file = join(func_indir, "slicetiming.txt")
        fsf = mrp.generate_fsf(func_image, smri_BET, MNI_2mm, save_dir=func_outdir, TR=TR,
                               slicetiming_mode=4, slicetiming_file=slice_timing_file,
                               delete_volumes=5, spatial_smoothing=4)
        # Run FEAT analysis
        feat_dir = mrp.FEAT(fsf, remove_reg_files=True,
                            force=FORCE_REGENERATE)
        if not os.path.exists(join(feat_dir, "filtered_func_data.nii.gz")):
            raise (FileNotFoundError(
                "filtered_func_data.nii.gz file was not found, FEAT analysis was probably not successful."))

        highres2standard = join(func_outdir, ".feat",
                                "reg", "highres2standard.nii.gz")
        highres2standard_moved = join(anat_outdir, "highres2standard.nii.gz")
        if FORCE_REGENERATE:
            if os.path.exists(highres2standard_moved):
                os.remove(highres2standard_moved)
            shutil.move(highres2standard, highres2standard_moved)
        else:
            if os.path.exists(highres2standard):
                shutil.move(highres2standard, highres2standard_moved)
            elif not os.path.exists(highres2standard_moved):
                raise (FileNotFoundError(
                    f"highres2standard.nii.gz not found in {anat_outdir}"))

    except Exception as e:
        logger.error(f"{ID}/{session}, ERROR IN FEAT:\n" + str(e))
        return progress
    progress["FEAT"] = True
    ##################################
    logger.info(f"{ID}/{session}:TIMESERIES EXTRACTION...")
    try:
        preprocessed_fmri = join(feat_dir, "filtered_func_data.nii.gz")
        preprocessed_fmri_example = join(feat_dir, "example_func.nii.gz")
        stan2func_matrix = join(feat_dir, "reg", "standard2example_func.mat")
        func2stan_matrix = join(feat_dir, "reg", "example_func2standard.mat")

        filtered_func2stan = join(
            func_outdir, "filtered_func2stan")
        timeseries_csv_path = join(
            TIMESERIES_SAVE_DIR, f"{ID}_{session}_timeseries.csv")
        if not os.path.exists(timeseries_csv_path) or FORCE_REGENERATE:
            timeseries_all = pd.DataFrame(columns=ATLAS_LABELS)
            if TIMESERIES_EXTRACTION_METHOD == 1:
                filtered_func2stan = mrp.apply_affine_transform(in_file=preprocessed_fmri,
                                                                matrix=func2stan_matrix,
                                                                ref=MNI_2mm,
                                                                out_file=filtered_func2stan,
                                                                force=FORCE_REGENERATE)

                roi_extracted_outdir = join(func_outdir, "roi_extrated")
                Path(roi_extracted_outdir).mkdir(
                    parents=True, exist_ok=True)

                for mask, label in zip(ATLAS_MASKS, ATLAS_LABELS):

                    out = join(roi_extracted_outdir,
                               f"roi_extracted_{label}.txt")
                    timeseries = mrp.get_mean_timeseries(filtered_func2stan,
                                                         mask,
                                                         out_file=out,
                                                         force=FORCE_REGENERATE)
                    timeseries_all[label] = timeseries
                if REMOVE_TRANSFERRED_FMRI_FILES:
                    os.remove(filtered_func2stan)

            elif TIMESERIES_EXTRACTION_METHOD == 2:
                roi_masks_outdir = join(func_outdir, "roi_masks")
                Path(roi_masks_outdir).mkdir(
                    parents=True, exist_ok=True)
                roi_extracted_outdir = join(func_outdir, "roi_extrated")
                Path(roi_extracted_outdir).mkdir(
                    parents=True, exist_ok=True)

                for mask, label in zip(ATLAS_MASKS, ATLAS_LABELS):
                    new_mask = mrp.apply_matrix_to_mask(mask,
                                                        label,
                                                        stan2func_matrix,
                                                        preprocessed_fmri_example,
                                                        roi_masks_outdir,
                                                        force=FORCE_REGENERATE)
                    out = join(roi_extracted_outdir,
                               f"roi_extracted_{label}.txt")
                    timeseries = mrp.get_mean_timeseries(preprocessed_fmri,
                                                         new_mask, out_file=out,
                                                         force=FORCE_REGENERATE)
                    timeseries_all[label] = timeseries
            elif TIMESERIES_EXTRACTION_METHOD == 3:
                splitted_fmri_dir = join(func_outdir, "splitted")
                Path(splitted_fmri_dir).mkdir(
                    parents=True, exist_ok=True)
                splitted_preprocesssed_fmri = mrp.split_fmri(preprocessed_fmri,
                                                             join(splitted_fmri_dir, "filtered_func"))
                splitted_preprocesssed_fmri2stan = []
                for v, volume in enumerate(splitted_preprocesssed_fmri):
                    volume2stan = mrp.apply_affine_transform(in_file=volume,
                                                             matrix=func2stan_matrix,
                                                             ref=MNI_2mm,
                                                             out_file=join(
                                                                 splitted_fmri_dir, f"filtered_func2stan{v:04}.nii.gz"),
                                                             force=FORCE_REGENERATE)
                    splitted_preprocesssed_fmri2stan.append(volume2stan)
                timeseries_all = mrp.get_mean_timeseries_py(splitted_preprocesssed_fmri2stan,
                                                            ATLAS)
                if REMOVE_TRANSFERRED_FMRI_FILES:
                    shutil.rmtree(splitted_fmri_dir)

            timeseries_all.to_csv(timeseries_csv_path, index=False)
        progress["timeseries"] = True

    except Exception as e:
        logger.error(
            f"{ID}/{session}, ERROR IN TIMESERIES EXTRACTION:\n" + str(e))
        return progress

    ##################################
    elapsed_time = time() - t_start
    logger.info(f"{ID}/{session} FINISHED. PROCESS TOOK {elapsed_time:.2f}s")
    progress["finish"] = True
    return progress


if __name__ == "__main__":
    t_start = time()
    subjects = glob.glob(join(DATASET_DIR, "**/**"))
    logger = create_logger(logs_path)
    with mp.Pool(mp.cpu_count()) as p:
        progresses = list(tqdm(p.imap_unordered(
            main, subjects), total=len(subjects)))
    df = pd.DataFrame(progresses)
    sums = df.sum(numeric_only=True, axis=0)
    total = df.shape[0]
    logger.info(
        f"""Final stats:
         init: {sums[0]}/{total}
         nifti merge: {sums[1]}/{total}
         BET: {sums[2]}/{total}
         .fsf: {sums[3]}/{total}
         FEAT: {sums[4]}/{total}
         timeseries:{sums[5]}/{total}
         finish:{sums[6]}/{total}
         invalid:{sums[7]}/{total}
         """)

    df.to_csv(join(SAVE_DIR, f"progresses_{time_of_start_fstr}.csv"), index=False)

    elapsed_time = time() - t_start
    logger.info(
        f"Took {elapsed_time:.2f}s in total, an average of {(elapsed_time/total):.2f}s per session")
