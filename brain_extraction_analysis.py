import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
"""
This script is used to find the average range of number of brain voxels in
a dataset.

It's advised to first run BET for all scans and after quality control
only use the good

"""

quality = pd.read_csv("/box/OrgData.csv")
info_df = pd.read_csv("/box/outputs/info.csv")


def main(path):
    indir = Path(path)
    session = indir.parts[-3]
    ID = indir.parts[-4]
    qc = int(quality.loc[(quality["Subject ID"] == ID) & (
        quality["VISCODE"] == session)]["Brain Extraction"])
    img = nib.load(path)
    voxels = img.get_fdata()
    ratio = np.count_nonzero(voxels) / voxels.size
    same_subject = info_df.loc[info_df["Subject"] == ID]
    if len(same_subject):
        sex = same_subject.iloc[0]["Sex"]
    else:
        sex = "?"
    return {"ID": ID, "session": session, "ratio": ratio, "qc": qc, "sex": sex}


if __name__ == "__main__":
    # BET_scans_paths = glob(
    #     "/box/outputs/preprocessed/**/**/anat/T1_BET_*_brain.nii.gz")

    # with mp.Pool() as p:
    #     result = list(tqdm(p.imap_unordered(
    #         main, BET_scans_paths), total=len(BET_scans_paths)))

    # df = pd.DataFrame(result)
    # print(df.columns)
    # df = df.dropna()
    # df.to_csv("brain_extraction_analysis.csv")
    df = pd.read_csv("brain_extraction_analysis.csv")

    ratio_range = tuple([df["ratio"].min(), df["ratio"].max()])

    male_df = df.loc[df["sex"] == "M"]
    male_df.groupby("qc")["ratio"].hist(
        stacked=True, bins=40, range=ratio_range)
    plt.savefig("brain_extraction_analysis_hist_M.png")

    female_df = df.loc[df["sex"] == "F"]
    female_df.groupby("qc")["ratio"].hist(
        stacked=True, bins=40, range=ratio_range)
    plt.savefig("brain_extraction_analysis_hist_F.png")
