import os
from os.path import join
import pathlib
import shutil
from time import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import preprocessing as mrp
from Utils.utils import create_logger
import pandas as pd
import glob
from datetime import datetime
from textwrap import dedent
# If set True, all process will redo all processes.
# If set False, it will only do the processes that were not successful or haven't been done yet.
FORCE_REGENERATE = False

MASKS = r"/box/outputs/masks"
ATLAS_MASKS, ATLAS_LABELS = mrp.create_masks_from_atlas(r"/box/AAL/ROI_MNI_V5.nii",
                                                        out_dir=MASKS, force=FORCE_REGENERATE)
STANDARDS = "/root/fsl/data/standard"
MNI_1mm = join(STANDARDS, "MNI152_T1_1mm_brain.nii.gz")
MNI_1mm_mask = join(STANDARDS, "MNI152_T1_1mm_brain_mask.nii.gz")

ADNI = "/box/test_dataset/"
OUT_DIR = "/box/outputs/preprocessed"
TIMESERIES_SAVE_PATH = "/box/outputs/timeseries"
pathlib.Path(TIMESERIES_SAVE_PATH).mkdir(exist_ok=True, parents=True)

t = datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
logs_path = f"/box/outputs/log_{t}.log"

REPETITION_TIME_TABLE_CSV_PATH = "/box/fmri_scans_repetitiontimes_TR.csv"
tr_csv = pd.read_csv(REPETITION_TIME_TABLE_CSV_PATH)


def main(indir):
    logger = create_logger(logs_path)

    t_start = time()
    indir = pathlib.Path(indir)
    session = indir.parts[-1]
    ID = indir.parts[-2]

    progress = {
        "ID": ID,
        "session": session,
        "init": False,
        "nifti merge": False,
        "BET": False,
        ".fsf": False,
        "FEAT": False,
        "mask application": False,
        "timeseries": False,
        "finish": False,
        "invalid": False,
    }
    session_outdir = join(OUT_DIR, ID, session)

    logger.info(f"{ID}/{session}:STARTED...")

    # Create a 'func' and 'anat' folder for each session for saving intermidiete files
    func_outdir = join(session_outdir, "func")
    pathlib.Path(func_outdir).mkdir(parents=True, exist_ok=True)
    anat_outdir = join(session_outdir, "anat")
    pathlib.Path(anat_outdir).mkdir(parents=True, exist_ok=True)

    func_indir = join(indir, "func")
    anat_indir = join(indir, "anat")

    progress["init"] = True
    ##################################
    TR = float(tr_csv.loc[(tr_csv["Subject ID"] == ID) &
                          (tr_csv["VISCODE"] == session)]["REPETITIONTIME"])
    fmri_frame_count = len(glob.glob(join(func_indir, "*.nii")))
    if fmri_frame_count > 210 or fmri_frame_count < 50:
        logger.error(
            f"{ID}/{session}:SCAN FLAGGED AS INVALID FOR HAVING TOO MANY OR TOO FEW FRAMES. N={fmri_frame_count}")
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
        fsf = mrp.generate_fsf(func_image, smri_BET, MNI_1mm, save_dir=func_outdir, TR=TR,
                               slicetiming_mode=4, slicetiming_file=slice_timing_file,
                               delete_volumes=5, spatial_smoothing=4)
        progress[".fsf"] = True
        # Run FEAT analysis
        feat_dir = mrp.FEAT(fsf, remove_reg_files=False,
                            force=FORCE_REGENERATE)
    except Exception as e:
        logger.error(f"{ID}/{session}, ERROR IN FEAT:\n" + str(e))
        return progress
    progress["FEAT"] = True
    ##################################
    logger.info(f"{ID}/{session}:MASK APPLICATION...")
    try:
        preprocessed_fmri = join(feat_dir, "filtered_func_data.nii.gz")
        preprocessed_fmri_example = join(feat_dir, "example_func.nii.gz")
        stan2func_matrix = join(feat_dir, "reg", "standard2example_func.mat")

        roi_masks_outdir = join(func_outdir, "roi_masks")
        pathlib.Path(roi_masks_outdir).mkdir(
            parents=True, exist_ok=True)
        roi_extracted_outdir = join(func_outdir, "roi_extrated")
        pathlib.Path(roi_extracted_outdir).mkdir(
            parents=True, exist_ok=True)

        timeseries_all = pd.DataFrame(columns=ATLAS_LABELS)
        for mask, label in zip(ATLAS_MASKS, ATLAS_LABELS):
            new_mask = mrp.apply_matrix_to_mask(mask,
                                                label,
                                                stan2func_matrix,
                                                preprocessed_fmri_example,
                                                roi_masks_outdir,
                                                force=FORCE_REGENERATE)
            out = join(roi_extracted_outdir, f"roi_extracted_{label}.txt")
            timeseries = mrp.get_mean_timeseries(preprocessed_fmri,
                                                 new_mask, out_file=out,
                                                 force=FORCE_REGENERATE)
            timeseries_all[label] = timeseries
        progress["mask application"] = True

        timeseries_all.to_csv(
            join(TIMESERIES_SAVE_PATH, f"{ID}_{session}_timeseries.csv"))
    except Exception as e:
        logger.error(
            f"{ID}/{session}:ERROR IN MASK APPLICATION:\n{label}:" + str(e))
        return progress

    progress["timeseries"] = True
    ##################################
    time_elapsed = time() - t_start
    logger.info(f"{ID}/{session}:FINISHED. PROCESS TOOK {time_elapsed:.2f}s")
    progress["finish"] = True
    return progress


if __name__ == "__main__":
    t1 = time()
    subjects = glob.glob(join(ADNI, "002_S_0413/m162"))
    logger = create_logger(logs_path)

    with mp.Pool(mp.cpu_count()) as p:
        progresses = list(tqdm(p.imap(main, subjects), total=len(subjects)))
    df = pd.DataFrame(progresses)
    sums = df.sum(numeric_only=True, axis=0)
    total = df.shape[0]
    logger.info(
        dedent(f"""Final stats:
	init: {sums[0]}/{total}
	nifti merge: {sums[1]}/{total}
	BET: {sums[2]}/{total}
	.fsf: {sums[3]}/{total}
	FEAT: {sums[4]}/{total}
	mask application:{sums[5]}/{total}
	timeseries:{sums[6]}/{total}
	finish:{sums[7]}/{total}
	invalid:{sums[8]}/{total}
	"""))
    df.to_csv(f"/box/outputs/progresses_{t1}.csv")

    t2 = time() - t1
    logger.info(
        f"Took {t2:.2f}s in total, an average of {(t2/len(subjects)):.2f}s per session")
