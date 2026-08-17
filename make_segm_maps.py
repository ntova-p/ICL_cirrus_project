import os
import uuid
import numpy as np
from astropy.io import fits
from detection import run as run_sextractor


input_file = "mini_training_set_tot_cir.fits"
out_dir = "mini_training_segm_maps_cir"

tmp_dir = "/tmp"
os.makedirs(out_dir, exist_ok=True)

OPTIONS = dict(
    DETECT_THRESH=2.5,
    DETECT_MINAREA=3,
    BACK_SIZE=5,
    DEBLEND_NTHRESH=32,
    DEBLEND_MINCONT=0.001,
    CHECKIMAGE_TYPE="SEGMENTATION"
)

hdul = fits.open(input_file)
hdu_map = {hdu.name: hdu for hdu in hdul if hdu.data is not None}

pairs = []

for name in hdu_map.keys():
    if name.endswith("_FINAL"):
        target = name.replace("_FINAL", "_TARGET")
        if target in hdu_map:
            pairs.append((name, target))

print(f"Found {len(pairs)} FINAL/TARGET pairs")

for final_name, target_name in pairs:

    final_hdu = hdu_map[final_name]
    target_hdu = hdu_map[target_name]

    print(f"\nProcessing: {final_name}")

    final_data = np.nan_to_num(final_hdu.data, nan=1e-32, posinf=1e-32, neginf=1e-32)
    final_data = np.clip(final_data, 1e-32, None).astype(np.float32)
    target_data = np.nan_to_num(target_hdu.data,nan=1e-32,posinf=1e-32,neginf=1e-32)
    target_data = np.clip(target_data, 1e-32, None).astype(np.float32)

    uid = uuid.uuid4().hex

    input_path = os.path.join(tmp_dir, f"{final_name}_{uid}.fits")
    seg_path = os.path.join(tmp_dir,f"seg_{uid}.fits")
    cat_path = os.path.join(tmp_dir, f"cat_{uid}.cat")

    fits.writeto(input_path, final_data, overwrite=True)

    run_sextractor(
        input_path,
        catalog_path=cat_path,
        run_label=uid,
        CHECKIMAGE_NAME=seg_path,
        **OPTIONS
    )

    seg = fits.getdata(seg_path)

    if seg.shape != final_data.shape:
        raise RuntimeError(f"Shape mismatch for {final_name}")

    mask = seg > 0
    seg_name = final_name.replace("_FINAL", "_SEGM")
    seg_out = os.path.join(out_dir,f"{seg_name}.fits")

    fits.writeto(seg_out,seg.astype(np.int32),header=final_hdu.header,overwrite=True)
    print(f"Saved segmentation map: {seg_out}")

    for f in [input_path, seg_path, cat_path]:
        if os.path.exists(f):
            os.remove(f)

print("\nDONE")