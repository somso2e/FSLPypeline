import os
from os.path import join
import pathlib
import shutil
from time import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import preprocessing as mrp
from Utils import create_logger
import pandas as pd
import glob
from datetime import datetime
import nibabel as nib
OUT_DIR = "/box/outputs/preprocessed"
time_of_star_f = datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
logs_path = f"/box/outputs/log_{time_of_star_f}.log"

AVERAGE_BRAIN_PERCENTAGE = 0.1


def main(indir):
    logger = create_logger(logs_path)

    t_start = time()
    indir = pathlib.Path(indir)
    session = indir.parts[-1]
    ID = indir.parts[-2]

    session_outdir = join(OUT_DIR, ID, session)

    anat_outdir = join(session_outdir, "anat")
    pathlib.Path(anat_outdir).mkdir(parents=True, exist_ok=True)

    anat_indir = join(indir, "anat")

    anat_image = (glob.glob(anat_indir + "/*.nii") +
                  glob.glob(anat_indir + "/*.nii.gz"))[0]
    try:
        smri_BET = mrp.brain_extraction(anat_image,
                                        join(anat_outdir,
                                             f"T1_BET_{ID}_brain.nii.gz"),
                                        force=True)
        voxels = nib.load(smri_BET).get_fdata()
        voxels_counts = np.count_nonzero(voxels)
        brain_percentage = voxels_counts / voxels.size
        brain_percentage_delta = brain_percentage - AVERAGE_BRAIN_PERCENTAGE
        frac = 0.5
        if brain_percentage_delta < -0.05:
            frac = 0.52
        elif -0.05 < brain_percentage_delta and brain_percentage_delta < 0:
            frac = 0.51
        elif 0 < brain_percentage_delta and brain_percentage_delta < 0.05:
            frac = 0.49
        elif brain_percentage_delta > 0.05:
            frac = 0.48

    except Exception as e:
        logger.error(f"{ID}/{session}:ERROR IN BET:\n" + str(e))
