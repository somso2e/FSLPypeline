from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from os.path import join
from PIL import ImageDraw, ImageFont


def concatenate_images(pics, resize=False):
    imgs = [Image.open(pic) for pic in pics]
    widths = [img.size[0] for img in imgs]
    w = max(widths)
    if resize:
        resized_imgs = []
        for img in imgs:
            wpercent = (w / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            resized_imgs.append(img.resize((w, hsize), Image.Resampling.LANCZOS))
        imgs = resized_imgs

    heights = [img.size[1] for img in imgs]
    h = np.sum(heights)

    out = Image.new("RGBA", (w, h))
    for i, img in enumerate(imgs):

        loc = ((w - img.size[0]) // 2, np.sum(heights[:i], dtype=int))
        out.paste(img, loc)
    return out


def bbox2_3D(img):
    all_axis = [(1, 2), (0, 2), (0, 1)]
    return [np.where(np.any(img, axis=axis))[0][[0, -1]] for axis in all_axis]


def reg_plots(feat_dir, save_dir):
    arial = ImageFont.truetype("./res/fonts/arial.ttf", 20)

    feat_dir = Path(feat_dir)
    session = feat_dir.parts[-3]
    ID = feat_dir.parts[-4]

    # Merge registration images
    reg_images = []
    REG_IMAGE_NAMES = ["example_func2highres.png",
                       "example_func2standard1.png", "highres2standard.png"]
    for img_name in REG_IMAGE_NAMES:
        img_path = join(feat_dir, "reg", img_name)
        if os.path.exists(img_path):
            reg_images.append(img_path)
        else:
            raise FileNotFoundError(f"{ID}/{session}: reg image img_name not found")

    reg_concatenated = concatenate_images(reg_images, resize=True)
    reg_concatenated.save(join(save_dir, f"{ID}_{session}_reg.png"))

    # Merge motion correction graphs
    mc_images = []
    MC_IMAGE_NAMES = ["disp.png", "rot.png", "trans.png"]
    for img_name in MC_IMAGE_NAMES:
        img_path = join(feat_dir, "mc", img_name)
        if os.path.exists(img_path):
            mc_images.append(img_path)
        else:
            raise FileNotFoundError(f"{ID}/{session}: mc image {img_name} not found")

    mc_concatenated = concatenate_images(mc_images, resize=True)
    # Write the voxel size on the picture
    if not os.path.exists(f"{feat_dir}/filtered_func_data.nii.gz"):
        raise FileNotFoundError(f"{ID}/{session}:filtered_func_data.nii.gz does not exists")

    cmd = f"fslinfo {feat_dir}/filtered_func_data.nii.gz"
    cmd_out = os.popen(cmd).read().split()

    voxel_size = f"voxel size: x={float(cmd_out[13]):.4f}, y={float(cmd_out[15]):.4f}, z={float(cmd_out[17]):.4f}"

    with open(join(feat_dir, "mc", "prefiltered_func_data_mcf_abs_mean.rms"), "r") as f:
        abs_mean_rms = float(f.read())
    with open(join(feat_dir, "mc", "prefiltered_func_data_mcf_rel_mean.rms"), "r") as f:
        rel_mean_rms = float(f.read())

    text = voxel_size + f", abs={abs_mean_rms:.4f}, rel={rel_mean_rms:.4f}"
    draw = ImageDraw.Draw(mc_concatenated)
    l, t, r, b = arial.getbbox(text)
    w = r - l
    h = b - t

    draw.text(((mc_concatenated.size[0] - w) / 2, (mc_concatenated.size[1] - h * 1.5)),
              text, font=arial, fill="black")
    mc_concatenated.save(os.path.join(
        save_dir, f"{ID}_{session}_mc.png"))


def brain_extraction_images(original_smri_file, brain_extracted_smri_file, save_path):
    """
    This function overlaps the highlighted brain extracted MRI scan
    on to the original scan on all 3 axis and on multiple slices.
    """
    plt.style.use('dark_background')
    plt.tight_layout()

    WHITE = [255, 255, 255, 0]
    MAGENTA = [255, 0, 255, 0]

    original_smri = nib.load(original_smri_file)
    brain_extracted_smri = nib.load(brain_extracted_smri_file)

    # Get the bounding box of the image
    bboxes = bbox2_3D(brain_extracted_smri.get_fdata())

    SLICES = [i / 6 for i in range(1, 6)]

    fig, axs = plt.subplots(3, len(SLICES), dpi=250)
    for plane, bbox in zip(range(3), bboxes):
        for s, slice in enumerate(SLICES):
            for smri, shade_color in zip([original_smri, brain_extracted_smri], [WHITE, MAGENTA]):
                smri_voxels = np.array(smri.get_fdata())
                smri_voxels = smri_voxels.swapaxes(0, plane)
                # normalize  data
                smri_voxels *= 255.0 / smri_voxels.max()

                first_slice, last_slice = bbox

                smri_slice = smri_voxels[int(
                    slice * (last_slice - first_slice) + first_slice)]
                smri_slice = smri_slice.reshape(
                    smri_slice.shape[0], smri_slice.shape[1])
                # make an RGBA picture with the RGB values set as the shade_color...
                rgba_canvas = np.full(
                    smri_slice.shape + (4,), shade_color, dtype=np.uint8)
                # ...and MRI values as the alpha values of the pixels
                rgba_canvas[:, :, 3] = smri_slice

                axs[plane, s].imshow(np.rot90(rgba_canvas))
            axs[plane, s].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()
