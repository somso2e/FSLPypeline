import pandas as pd
import glob
import os
from PIL import ImageDraw, ImageFont
from os.path import join
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import preprocessing as mrp

arial = ImageFont.truetype("./res/fonts/arial.ttf", 20)

PREPROCESSED_OUTDIR = "/box/outputs/preprocessed"

SAVE_DIR = "/box/outputs/reg_pics_feat/"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


def quality_control(feat_dir):
    mrp.reg_plots(feat_dir, SAVE_DIR)
    try:
        original_smri_file = glob.glob(
            f"/box/ADNI/{ID}/{session}/anat/*.nii")[0]
        brain_extracted_smri_file = glob.glob(os.path.join(
            PREPROCESSED_OUTDIR, f"{ID}/{session}/anat/T1_BET_*.nii.gz"))[0]
    except IndexError:
        tqdm.write(f"{ID}/{session} BET or original smri scan not found.")
        return

    mrp.brain_extraction_images(original_smri_file,
                                brain_extracted_smri_file,
                                os.path.join(SAVE_DIR, f"{ID}_{session}_BET.png"))


bet_qc = pd.read_csv(
    "/box/outputs/final_BET_result.csv", index_col=False)

if __name__ == "__main__":
    FEAT_DIRS = glob.glob(os.path.join(
        PREPROCESSED_OUTDIR, "**/**/func/.feat/"))
    FEAT_DIRS = []
    for i, row in bet_qc.iterrows():
        FEAT_DIRS.append(
            join(PREPROCESSED_OUTDIR, row["ID"], row["session"], "func", ".feat"))
    # FEAT_DIRS=FEAT_DIRS[:2]
    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm(p.imap_unordered(
            quality_control, FEAT_DIRS), total=len(FEAT_DIRS)))
