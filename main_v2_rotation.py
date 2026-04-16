# -*- coding: utf-8 -*-


import numpy as np
import galsim
import os
import logging
from astropy.io import fits
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from astropy import wcs
from scipy.special import gammaincinv
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from scipy.special import gamma


from galaxy_class import Galaxy
from icl import ICL
from utils import add_stamp_to_image, weighted_distribution_ellipse, find_BCG, randomize_theta_dataframe, to_pixel_general



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)




def main(halo_id):

   #  halo_id = 4138390381366

    DATA_DIR = Path("/data2/ntova/cluster_init_data/")
    data_file = f'{halo_id}_data.fits'
    hdul = fits.open(DATA_DIR / data_file)


    ## ANALYTICAL/IMAGE PARAMETERS
    pixel_scale = 2
    px_scale = pixel_scale
    folding_threshold = 5e-3
    maximum_fft_size = 32768

    ## FIXED PARAMETERS FOR ALL GALACTIC HALOS
    gh_flux_factor = 0.1        # how does each galactic halo flux scale with the galaxy's flux
    halo_n = 0.8                # galactic halo sersic idx

    ## ICL PARAMETERS
    mu0_ICL = 20.0
    mu0_icl = mu0_ICL
    zp_flagship = -48.6
    n_icl = 4                  # for initial tests fixed at n=4, then will be a free parameter

    ## NOISE PARAMETERS
    zp_euclid = 30
    mu_lim = 29



    
    data = hdul[1].data
    data_header = hdul[1].header
    table_hdu = hdul[1]
    cols = table_hdu.columns
    hdul.close()

    full_loc_cut = data
    idx = np.argsort(full_loc_cut['euclid_vis'])[::-1]
    full_loc_cut = full_loc_cut[idx]


    ## in case we want to perform a mass cut to exclude small galaxies ---- we will not at the moment
    
    min_gal_mass = 8
    loc_cut = full_loc_cut[ full_loc_cut['log_stellar_mass'] >= min_gal_mass]

    USE_MASS_CUT = False

    if USE_MASS_CUT:
        catalog = loc_cut
    else:
        catalog = full_loc_cut


    ## CLUSTER PARAMETERS
    M = data[10]['lm_halo']
    z = data[10]['true_redshift_halo']





    #### BCG
    bcg_idx = find_BCG(catalog)
    bcg_ra = catalog[bcg_idx]['ra_gal']    
    bcg_dec = catalog[bcg_idx]['dec_gal']  


    bcg = Galaxy(catalog[bcg_idx], gh_flux_factor, folding_threshold, maximum_fft_size, halo_n)

    ###### COORDINATES ########

    ra_min = np.min(catalog['ra_gal'])
    ra_max = np.max(catalog['ra_gal'])
    dec_min = np.min(catalog['dec_gal'])
    dec_max = np.max(catalog['dec_gal'])

    pixel_scale_arcsec = pixel_scale
    pixel_scale_deg = pixel_scale_arcsec / 3600.0

    width_deg  = ra_max  - ra_min
    height_deg = dec_max - dec_min

    nx, ny = 2000,2000

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nx/2, ny/2]                                   # reference pixel = center
    w.wcs.crval = [(ra_min+ra_max)/2, (dec_min+dec_max)/2]       # world value at center
    w.wcs.cdelt = np.array([-pixel_scale_deg, pixel_scale_deg])  # RA decreases with x
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]







    ###### IMAGE GENERATION ######

    
    i = len(catalog)
    flux_fraction = 0.99
    save_every = 500

    #image_galaxies = galsim.ImageF(nx,ny, scale=pixel_scale, init_value=0) #image_size, image_size,
    #image_gal_halos = galsim.ImageF(nx,ny, scale=pixel_scale, init_value=0)
    #image_ICL = galsim.ImageF(nx,ny, scale=pixel_scale, init_value=0)



    ###### randomized version

    num_versions = 5  # 1 original + 4 random realizations  
    image_gal_list = [galsim.ImageF(nx, ny, scale=pixel_scale, init_value=0) for _ in range(num_versions)]  
    image_gal_halos_list = [galsim.ImageF(nx, ny, scale=pixel_scale, init_value=0) for _ in range(num_versions)]
    max_angles_list = [30, 60, 90, 180]

    ra_array = catalog['ra_gal']
    dec_array = catalog['dec_gal']
    ra_random_list = []
    dec_random_list = []

    for max_angle in max_angles_list:
        ra_r, dec_r = randomize_theta_dataframe(ra_array, dec_array, bcg_ra, bcg_dec, max_angle=max_angle)
        ra_random_list.append(ra_r)
        dec_random_list.append(dec_r)



    ####### GALAXIES #######
    
    for idx, row in enumerate(catalog[:i]):
        g = Galaxy(row, gh_flux_factor, folding_threshold, maximum_fft_size, halo_n)
        x = y = g.adaptive_stamp_size(pixel_scale=pixel_scale, fraction=flux_fraction)
        g_sim = g.make_galsim_galaxy(scale_up=1)


        if idx < 7:

            try:
                stamp = g_sim.drawImage(nx=x, ny=y, scale=pixel_scale, method='fft')
            except galsim.GalSimError as e_fft:
                print(f"FFT failed for galaxy {idx+1}: {e_fft}. Trying real-space...")
                
                try:
                    stamp = g_sim.drawImage(nx=x, ny=y, scale=pixel_scale, method='real_space')
                except galsim.GalSimError as e_real:
                    print(f"Real-space also failed for galaxy {idx+1}: {e_real}. Skipping.")
                    continue

        else:
            try:
                stamp = g_sim.drawImage(nx=x, ny=y, scale=pixel_scale, method='real_space')
            except galsim.GalSimError as e_real:
                logging.info(f"Real-space failed for galaxy {idx+1}: {e_real}. Trying FFT...")

                try:
                    stamp = g_sim.drawImage(nx=x, ny=y, scale=pixel_scale, method='fft')
                except galsim.GalSimError as e_fft:
                    logging.info(f"FFT also failed for galaxy {idx+1}: {e_fft}. Skipping.")
                    continue


        #x_pix, y_pix = g.to_pixel(w)
        add_stamp_to_image(image_gal_list[0], stamp, g.ra, g.dec, w=w)


        # extra images: place stamps at randomized RA/Dec
        for version in range(1, num_versions):
            ra_gal = ra_random_list[version-1][idx]
            dec_gal = dec_random_list[version-1][idx]
            x_pix, y_pix = to_pixel_general(w, ra_gal, dec_gal)

            add_stamp_to_image(image_gal_list[version], stamp, ra=ra_gal, dec=dec_gal, w=w)

        logging.info(f"Galaxy {idx+1} printed, stamp size: {x}x{y}, flux: {stamp.array.sum():.3e}")


    hdulist_gal = [fits.PrimaryHDU(image_gal_list[0].array)]  # original
    for img in image_gal_list[1:]:
        hdulist_gal.append(fits.ImageHDU(img.array))  

    hdul_gal = fits.HDUList(hdulist_gal)
    hdul_gal.writeto(f'{halo_id}_galaxies.fits', overwrite=True)
    
    
    ########### galactic halos ##########
    
    halo_max_mass = 11

    for idx, row in enumerate(catalog[:i]):
        g = Galaxy(row, gh_flux_factor, folding_threshold, maximum_fft_size, halo_n)
        x = y = g.adaptive_stamp_size(pixel_scale=pixel_scale, fraction=flux_fraction)
        g_sim = g.make_galsim_galaxy(scale_up=1)

        if g.log_stellar_mass>halo_max_mass:          # idx<8:
            h_sim = g.make_galsim_halo(halo_r50=None, scale_up=1)
            halo_x = halo_y = min(g.adaptive_halo_stamp_size(pixel_scale=pixel_scale, scale_up=1),512)

            halo_stamp = h_sim.drawImage(nx=halo_x, ny=halo_y, scale=pixel_scale, method='real_space')
            
            
            #add_stamp_to_image(image_gal_halos, halo_stamp, g, w)
            # original image
            #add_stamp_to_image(image_gal_halos_list[0], halo_stamp, g, w)
            add_stamp_to_image(image_gal_halos_list[0], halo_stamp, g.ra, g.dec, w)

            # extra images: place halos at the same randomized RA/Dec as galaxies
            for i in range(1, num_versions):
                ra_gal = ra_random_list[i-1][idx]       
                dec_gal = dec_random_list[i-1][idx]     
                add_stamp_to_image(image_gal_halos_list[i], halo_stamp, ra_gal, dec_gal, w)     
            
            print(f"HALO {idx} added: size {halo_x}x{halo_y}, flux={halo_stamp.array.sum():.3e}")

    #fits.writeto(f'{halo_id}_galhalos.fits', image_gal_halos.array, overwrite=True)
    
    hdulist_halo = [fits.PrimaryHDU(image_gal_halos_list[0].array)]  # original
    for img in image_gal_halos_list[1:]:
        hdulist_halo.append(fits.ImageHDU(img.array))

    hdul_halo = fits.HDUList(hdulist_halo)  
    hdul_halo.writeto(f'{halo_id}_galhalos.fits', overwrite=True)   
    


    ######## also write file with positions #######

    all_ra = [ra_array] + ra_random_list
    all_dec = [dec_array] + dec_random_list
    num_versions = len(all_ra)
    num_galaxies = len(ra_array)

    cols = []
    for i in range(num_versions):
        cols.append(fits.Column(name=f'RA_v{i}', format='D', array=all_ra[i]))
        cols.append(fits.Column(name=f'DEC_v{i}', format='D', array=all_dec[i]))

    hdu_table = fits.BinTableHDU.from_columns(cols)
    hdu_primary = fits.PrimaryHDU()
    hdul = fits.HDUList([hdu_primary, hdu_table])
    hdul.writeto(f'{halo_id}_galaxy_positions.fits', overwrite=True)





    ############ ICL ###########



    ##### calculating all sattelite distributions (also for randomized images)

    e_pa_array = np.zeros((num_versions, 2  ))  # column 0 = e, column 1 = PA_deg
        
    for v in range(num_versions):   
        if v == 0:  
            ra_use = catalog['ra_gal']  
            dec_use = catalog['dec_gal']    
        else:   
            ra_use = ra_random_list[v-1]    
            dec_use = dec_random_list[v-1]  


        bcg_coord = SkyCoord(ra=bcg.ra*u.deg, dec=bcg.dec*u.deg)
        all_coords = SkyCoord(ra=ra_use*u.deg, dec=dec_use*u.deg)
        R_phys = 1 * u.Mpc
        D_A = cosmo.angular_diameter_distance(bcg.redshift_gal)
        theta = (R_phys / D_A) * u.rad
        mask = bcg_coord.separation(all_coords) <= theta

        points = np.column_stack((ra_use[mask], dec_use[mask]))
        weights = catalog['log_luminosity_r01'][mask]

        result = weighted_distribution_ellipse(points, weights, nsigma=1.0)
        e_icl = result['ellipticity']
        pa_icl_deg = result['position_angle_deg']

        e_pa_array[v, 0] = e_icl
        e_pa_array[v, 1] = pa_icl_deg




    ##### plotting the ICL images ######

    icl_images = []


    rel_pa = 0    ## in deg --- set to 0 for the first tests, but will be a free parameter later
    n_icl = 4

    cluster_flux = np.sum(catalog['euclid_vis'])
    cluster_mass = 10**M


    for v in range(num_versions):
        e_icl = e_pa_array[v, 0]
        pa_icl_deg = e_pa_array[v, 1]

        icl = ICL(
            bcg=bcg,
            wcs=w,
            m200=cluster_mass,
            cluster_flux=cluster_flux,
            n_sersic=n_icl,
            e=e_icl,
            pa=pa_icl_deg + rel_pa,   # adjust by rel_pa if needed
        )

        Re_icl = icl.compute_re_from_mu0(mu0=mu0_icl, ZP=zp_flagship)
        icl.make_profile(mu0_icl, zp_flagship)

        icl_nx = icl_ny = icl.adaptive_stamp_size(
            pixel_scale=px_scale,
            flux_fraction=0.999,
            buffer_factor=2.0,
            max_pixels=2000
        )

        icl_stamp = icl.profile.drawImage(
            nx=icl_nx,
            ny=icl_ny,
            scale=px_scale,
            method="sb"
        )

        icl_images.append(icl_stamp.array)


    hdul_icl = fits.HDUList([fits.PrimaryHDU(icl_images[0])])
    for img_array in icl_images[1:]:
        hdul_icl.append(fits.ImageHDU(img_array))



    hdul_icl[0].header['VERSION'] = 0
    hdul_icl[0].header['E'] = e_pa_array[0, 0]
    hdul_icl[0].header['PA'] = e_pa_array[0, 1]

    for version, hdu in enumerate(hdul_icl[1:], start=1):
        hdu.header['VERSION'] = version
        hdu.header['E'] = e_pa_array[version, 0]
        hdu.header['PA'] = e_pa_array[version, 1]

    hdul_icl.writeto(f'{halo_id}_ICL_{n_icl}_{rel_pa}.fits', overwrite=True)












if __name__ == "__main__":

    halo_ids = [

        3935360166071,
        3936350335014,
        4138390381366,
        4040370681547,
        3939360766380,
        3935360398928,
        3839380916116,
        3740360694380
        



#            3740360529492,      
#            4037370014509,      
#            4242400187314,      
#            4036350332263,      
#            3545380349863,      
#            3840360729382,      
#            3540350196678,      
#            4240360271546,      
#            4338380622430,      
#            4036370818677,      
#            3740360074867,      
#            3939350664369,      
#            3939380821520,      
#            4039350321288

#            3943390113633,
#            4037350546841,      
#            3838370682076,      
#            3938350271698,      
#            3845360361913,      
#            3739350897596,      
#            4041370090393,      
#            3542370665932,      
#            3544360043075,      
#            4240360760992,      
#            4044360321731    

  #          3840370279423, 
  #          3842360376602, 
  #          4044390697084, 
  #          3842390652567, 
  #         3842390656079, 
  #          4237350881743, 
            









 #       3942350892355,
 #       3941350039409,


  #      3941350039409,
  #      3942350028873,
  #      3942350099797,
  #      3942350100585,
  #      3942350657043,
  #      3942350892355,
  #      3943350670278,
  #      3944350426869,
  #      3741350060877,
  #      4035350147650,
  #      4035350214007,
  #      4035350509680,
  #      4035350548318,
  #      4044350538380,
  #      4140350647736
    ]

    for hid in halo_ids:
        print(f"\nProcessing halo {hid}\n")
        main(hid)