from os.path import join
import pathlib
from time import time
from tqdm import tqdm
import multiprocessing as mp
import preprocessing as mrp
from Utils import create_logger
import pandas as pd
import glob
from datetime import datetime

ADNI = "/box/ADNI/"


OUT_DIR = "/box/outputs/preprocessed"
time_of_star_f = datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
logs_path = f"/box/outputs/log_brute_force_{time_of_star_f}.log"


SAVE_DIR = "/box/outputs/reg_pics_brute_forced_2/"
pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


def main(data):
    ID = data["ID"]
    session = data["session"]

    indir = join(ADNI, ID, session)

    logger = create_logger(logs_path)
    logger.info(f"{ID}/{session}:STARTED")

    session_outdir = join(OUT_DIR, ID, session)

    anat_outdir = join(session_outdir, "anat")
    pathlib.Path(anat_outdir).mkdir(parents=True, exist_ok=True)

    anat_indir = join(indir, "anat")
    anat_image = (glob.glob(anat_indir + "/*.nii") +
                  glob.glob(anat_indir + "/*.nii.gz"))[0]

    suggestion = data["suggestion"]
    if suggestion == "remove" or suggestion == "?" or suggestion == "؟" or suggestion == "rd1":
        logger.error(
            f"{ID}/{session}: Skipping based on suggestion: {suggestion}")
        return
    try:
        suggestion = int(suggestion)
        if data["rd"] == 0:
            if suggestion < 46:
                fracs_deltas = [-4, -2, 0]
            elif suggestion > 54:
                fracs_deltas = [0, 2, 4]
            else:
                fracs_deltas = [0]
            for frac_delta in fracs_deltas:
                frac = (suggestion + frac_delta) / 100
                smri_BET = mrp.brain_extraction(anat_image,
                                                join(
                                                    anat_outdir, f"T1_BET_{ID}_brain_f{str(frac)[2:]}_rd0.nii.gz"),
                                                frac=frac,
                                                force=True)
                mrp.brain_extraction_images(anat_image,
                                            smri_BET,
                                            join(SAVE_DIR, f"{ID}_{session}_BET_f{str(frac)[2:]}_rd0.png"))
                logger.info(f"{ID}/{session}:ROBUST,FRAC {frac} FINISHED")
        else:
            for frac_delta in [0]:
                frac = (suggestion + frac_delta) / 100
                smri_BET = mrp.brain_extraction(anat_image,
                                                join(
                                                    anat_outdir, f"T1_BET_{ID}_brain_f{frac}_rd1.nii.gz"),
                                                frac=frac,
                                                reduce_bias=True,
                                                robust=False,
                                                force=True)
                mrp.brain_extraction_images(anat_image,
                                            smri_BET,
                                            join(SAVE_DIR, f"{ID}_{session}_BET_f{frac}_rd1.png"))
                logger.info(f"{ID}/{session}:REDUCE BIAS,FRAC {frac} FINISHED")
    except Exception as e:
        logger.error(f"{ID}/{session}:ERROR ENCOUNTERED:\n" + str(e))


if __name__ == "__main__":
    t_start = time()
    df = pd.read_csv("/box/brute_force_qc_table_bads.csv")
    df = df.to_dict(orient='records')
    logger = create_logger(logs_path)
    with mp.Pool(mp.cpu_count()) as p:
        progresses = list(tqdm(p.imap_unordered(
            main, df), total=len(df)))

    elapsed_time = time() - t_start
    logger.info(
        f"Took {elapsed_time:.2f}s in total, an average of {(elapsed_time/len(df)):.2f}s per session")
