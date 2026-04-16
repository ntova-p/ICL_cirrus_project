import numpy as np
import matplotlib.pyplot as plt
import galsim
import sys
import os
import math
import logging
import pandas as pd
from astropy.io import fits
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from astropy import wcs
from PIL import Image
from scipy.special import gammaincinv
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
import matplotlib.patches as patches
from astropy.coordinates import SkyCoord
from scipy.special import gamma
from photutils.background import Background2D, SExtractorBackground, MedianBackground
from photutils.segmentation import detect_sources
from scipy import ndimage

from galaxy_class import Galaxy
from icl import ICL
from utils import weighted_distribution_ellipse, add_stamp_to_image, find_BCG


bg_mag =  22.3   # [mag arcsec-2]
lim_sb = 29.9    # [mag arcsec-2], 10''x10'' scale at 1sigma
zp_phot = 30.13   #photometric zeropoint
zp_fs= -48.6   # flagship zeropoint

threshold, boxsize = 5,8 # parameters for source masks
std_dev, m_thres = 1.5, 0.05 # extended mask params

pixel_scale = 2
f_bg_per_arcsec2 = 10**(-(bg_mag -zp_fs)/2.5)
f_bg_per_pixel = f_bg_per_arcsec2

def crop_center(array, crop_size=256):
    H, W = array.shape
    start_y = (H - crop_size) // 2
    start_x = (W - crop_size) // 2
    return array[start_y:start_y+crop_size, start_x:start_x+crop_size]

def original_img_correction(img, pixel_scale=2):
    img_new = (img*1e-27)*(pixel_scale**2)
    return img_new

def flux_to_mag(img, zp = zp_fs):
    mag_array = -2.5 * np.log10(img) + zp
    return mag_array


def random_cutout(arr, size=256):
    h = len(arr)
    w = len(arr[0])

    start_row = np.random.randint(0, h - size)
    start_col = np.random.randint(0, w - size)

    cutout = [row[start_col:start_col + size] for row in arr[start_row:start_row + size]]
    
    return cutout


def detect_and_mask_sources(image, threshold=3, box_size=32):

    data =image
    Bkg = Background2D(image, bkg_estimator=SExtractorBackground(), box_size=box_size)
    back = Bkg.background
    back_rms = Bkg.background_rms
    
    threshold = back + (threshold * back_rms)
    segm = detect_sources(data, threshold,  npixels=5)
    mask = segm.data>0

    return mask, back, segm

def grow_mask_gaussian(mask, stddev=std_dev, mask_thresh=m_thres):
    mask_grown = ndimage.gaussian_filter((mask > 0).astype(float), sigma=stddev)
    
    return mask_grown > mask_thresh











def process_halos_all_in_one(halo_ids, output_file='mini_training_set.fits'):
    hdulist = fits.HDUList([fits.PrimaryHDU()])  # Primary HDU empty

    for halo_idx, halo_id in enumerate(halo_ids):
        print(f"Processing halo {halo_id} ({halo_idx+1}/{len(halo_ids)})...")

        # Load cluster metadata
        data = fits.open('clusters_metadata.fits')[1].data
        cluster_info = data[data['halo_id'] == halo_id]
        M, z = cluster_info['lm_halo'], cluster_info['true_redshift_halo']

        # Open component FITS files
        cl_galaxies = fits.open(f'new_component_fits/{halo_id}_galaxies.fits')
        galhalos = fits.open(f'new_component_fits/{halo_id}_galhalos.fits')
        ICL = fits.open(f'new_component_fits/{halo_id}_ICL_4_0.fits')

        gal_img, halo_img, icl_img, tot_img = [], [], [], []

        for i in range(5):
            g_img = original_img_correction(cl_galaxies[i].data)
            h_img = original_img_correction(galhalos[i].data)
            i_img = original_img_correction(ICL[i].data)
            gal_img.append(g_img)
            halo_img.append(h_img)
            icl_img.append(i_img)

        for i in range(5):
            t_img = gal_img[i] + halo_img[i] + icl_img[i]
            tot_img.append(t_img)


        #flip and rotate
        img_fliplr = np.fliplr(np.copy(tot_img[0]))
        icl_fliplr = np.fliplr(np.copy(icl_img[0]))

        img_rot = np.rot90(np.copy(tot_img[1]), k=1)
        icl_rot = np.rot90(np.copy(icl_img[1]), k=1)

        img_flipud = np.copy(tot_img[2])[::-1]
        icl_flipud = np.copy(icl_img[2])[::-1]


        tot_img.append(img_fliplr)
        tot_img.append(img_rot)
        tot_img.append(img_flipud)

        icl_img.append(icl_fliplr)
        icl_img.append(icl_rot)
        icl_img.append(icl_flipud)

        # Random background galaxies
        hdul_bg = fits.open('galaxy_random_canvases_500.fits')
        n = len(hdul_bg) - 1
        rand_index = np.random.randint(1, n)
        bg_galaxies_or = original_img_correction(hdul_bg[rand_index].data)

        # Noise parameters
        f_1sigma = 10**(-0.4 * (lim_sb - zp_fs))
        #N_pix = (10 / pixel_scale)**2
        #sigma_per_px = f_1sigma / np.sqrt(N_pix)    
        sigma_per_px = f_1sigma * np.sqrt((10 * pixel_scale) ** 2)
        rng = galsim.BaseDeviate()
        gaussian_noise = galsim.GaussianNoise(rng, sigma=sigma_per_px)


        def add_noise_and_bg(cluster, bg_galaxies):
            img = cluster + random_cutout(bg_galaxies, 256)
            img = galsim.Image(img)
            img.addNoise(gaussian_noise)
            return img.array

        final_img, target_icl = [], []

        for i in range(len(tot_img)):
            fin = crop_center(tot_img[i], 256)
            final_img.append(add_noise_and_bg(fin, bg_galaxies_or))
            target_icl.append(crop_center(icl_img[i], 256))



        # Masked Images
        masked, masked_extended, masked_keep_bcg, masked_extended_keep_bcg = [], [], [], []
        mask_list, mask_grown_list, mask_keep_bcg_list, mask_grown_keep_bcg_list = [], [], [], []

        for i in range(len(tot_img)):

            img = final_img[i]

                        
            mask, bg, segm = detect_and_mask_sources(img, threshold=threshold, box_size=boxsize)
            segm.data[128,128]
            mask_keep_bcg = (segm.data>0) & (segm.data!=segm.data[128,128])
            mask_grown = grow_mask_gaussian(mask)
            mask_grown_keep_bcg = grow_mask_gaussian(mask_keep_bcg)


            masked_image = np.where(~mask, img, 1e-30) 
            masked_image_grown = np.where(~mask_grown, img, 1e-30)
            masked_image_keep_BCG = np.where(~mask_keep_bcg, img, 1e-30) 
            masked_image_grown_keep_BCG = np.where(~mask_grown_keep_bcg, img, 1e-30)

            #mask, bg = detect_and_mask_sources(img, threshold=threshold, box_size=boxsize)
            #mask_grown = grow_mask_gaussian(mask)
            #masked_image = np.where(~mask, img, 1e-30) 
            #masked_image_grown = np.where(~mask_grown, img, 1e-30)

            masked.append(masked_image)
            masked_extended.append(masked_image_grown)
            masked_keep_bcg.append(masked_image_keep_BCG)
            masked_extended_keep_bcg.append(masked_image_grown_keep_BCG)

            mask_list.append(mask.astype(np.float32))
            mask_grown_list.append(mask_grown.astype(np.float32))
            mask_keep_bcg_list.append(mask_keep_bcg.astype(np.float32))
            mask_grown_keep_bcg_list.append(mask_grown_keep_bcg.astype(np.float32))



        ## Clipping anything below detection limit
        final_img = [np.clip(img, 1e-30, None) for img in final_img]
        target_icl = [np.clip(img, 1e-30, None) for img in target_icl]
        masked = [np.clip(img, 1e-30, None) for img in masked]
        masked_extended = [np.clip(img, 1e-30, None) for img in masked_extended]
        masked_keep_bcg = [np.clip(img, 1e-30, None) for img in masked_keep_bcg]
        masked_extended_keep_bcg = [np.clip(img, 1e-30, None) for img in masked_extended_keep_bcg]

        # Append all doublets for this halo to the main HDU list
        for i in range(len(tot_img)):
            # Final image HDU
            hdu_final = fits.ImageHDU(final_img[i], name=f'H{halo_id}_D{i+1}_FINAL')
            hdu_final.header['HALO_ID'] = halo_id
            hdu_final.header['DOUBLET'] = i+1
            hdulist.append(hdu_final)

            # Target ICL HDU
            hdu_icl = fits.ImageHDU(target_icl[i], name=f'H{halo_id}_D{i+1}_ICL')
            hdu_icl.header['HALO_ID'] = halo_id
            hdu_icl.header['DOUBLET'] = i+1
            hdulist.append(hdu_icl)

            # Masked Images
            hdu_masked = fits.ImageHDU(masked[i], name=f'H{halo_id}_D{i+1}_MASKED')
            hdu_masked.header['HALO_ID'] = halo_id
            hdu_masked.header['DOUBLET'] = i+1
            hdu_masked.header['TS'] = threshold
            hdu_masked.header['BOXSIZE'] = boxsize
            hdulist.append(hdu_masked)

            hdu_masked_extended = fits.ImageHDU(masked_extended[i], name=f'H{halo_id}_D{i+1}_MASKED_EXTENDED')
            hdu_masked_extended.header['HALO_ID'] = halo_id
            hdu_masked_extended.header['DOUBLET'] = i+1
            hdu_masked_extended.header['TS'] = threshold
            hdu_masked_extended.header['BOXSIZE'] = boxsize
            hdu_masked_extended.header['STDDEV'] = std_dev
            hdu_masked_extended.header['MTS'] = m_thres
            hdulist.append(hdu_masked_extended)

            hdu_masked_keep_bcg = fits.ImageHDU(masked_keep_bcg[i], name=f'H{halo_id}_D{i+1}_MASKED_KEEP_BCG')
            hdu_masked_keep_bcg.header['HALO_ID'] = halo_id
            hdu_masked_keep_bcg.header['DOUBLET'] = i+1
            hdu_masked_keep_bcg.header['TS'] = threshold
            hdu_masked_keep_bcg.header['BOXSIZE'] = boxsize
            hdulist.append(hdu_masked_keep_bcg)

            hdu_masked_extended_keep_bcg = fits.ImageHDU(masked_extended_keep_bcg[i], name=f'H{halo_id}_D{i+1}_MASKED_EXTENDED_KEEP_BCG')
            hdu_masked_extended_keep_bcg.header['HALO_ID'] = halo_id
            hdu_masked_extended_keep_bcg.header['DOUBLET'] = i+1
            hdu_masked_extended_keep_bcg.header['TS'] = threshold
            hdu_masked_extended_keep_bcg.header['BOXSIZE'] = boxsize
            hdu_masked_extended_keep_bcg.header['STDDEV'] = std_dev
            hdu_masked_extended_keep_bcg.header['MTS'] = m_thres
            hdulist.append(hdu_masked_extended_keep_bcg)

            # --- Raw Masks ---
            hdu_mask = fits.ImageHDU(mask_list[i], name=f'H{halo_id}_D{i+1}_MASK')
            hdu_mask.header['HALO_ID'] = halo_id
            hdu_mask.header['DOUBLET'] = i+1
            hdulist.append(hdu_mask)

            hdu_mask_grown = fits.ImageHDU(mask_grown_list[i], name=f'H{halo_id}_D{i+1}_MASK_GROWN')
            hdu_mask_grown.header['HALO_ID'] = halo_id
            hdu_mask_grown.header['DOUBLET'] = i+1
            hdu_mask_grown.header['STDDEV'] = std_dev
            hdu_mask_grown.header['MTS'] = m_thres
            hdulist.append(hdu_mask_grown)

            hdu_mask_keep_bcg = fits.ImageHDU(mask_keep_bcg_list[i], name=f'H{halo_id}_D{i+1}_MASK_KEEP_BCG')
            hdu_mask_keep_bcg.header['HALO_ID'] = halo_id
            hdu_mask_keep_bcg.header['DOUBLET'] = i+1
            hdulist.append(hdu_mask_keep_bcg)

            hdu_mask_grown_keep_bcg = fits.ImageHDU(mask_grown_keep_bcg_list[i], name=f'H{halo_id}_D{i+1}_MASK_GROWN_KEEP_BCG')
            hdu_mask_grown_keep_bcg.header['HALO_ID'] = halo_id
            hdu_mask_grown_keep_bcg.header['DOUBLET'] = i+1
            hdu_mask_grown_keep_bcg.header['STDDEV'] = std_dev
            hdu_mask_grown_keep_bcg.header['MTS'] = m_thres
            hdulist.append(hdu_mask_grown_keep_bcg)


    # Save all halos in one multi-extension FITS
    hdulist.writeto(output_file, overwrite=True)
    print(f"Saved all halos in a single FITS: {output_file}")


if __name__ == "__main__":   
    
    halo_ids = [3842390652567,
                4037350546841,
                3839380916116,
                4035350509680,
                3941350039409,
                3540350196678,
                4040370681547,
                3942350657043,
                3943350670278,
                4237350881743,
                3739350897596,
                3935360398928,
                3939380821520,
                4039350321288,
                3544360043075,
                3938350271698,
                4138390381366,
                4338380622430,
                3842360376602,
                3939360766380]
                    
    
    process_halos_all_in_one(halo_ids)

