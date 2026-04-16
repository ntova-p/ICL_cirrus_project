import numpy as np
import galsim
from scipy.special import gammaincinv



class Galaxy:

    def __init__(self, row, gh_flux_factor, folding_threshold, maximum_fft_size, halo_n):

        self.halo_id = row['halo_id']
        self.galaxy_id = row['galaxy_id']
        self.kind = row['kind']
        self.dominant_shape = row['dominant_shape']
        self.disk_angle = row['disk_angle']
        self.log_lum = row['log_luminosity_r01']
        self.euclid_vis = row['euclid_vis']
        self.log_stellar_mass = row['log_stellar_mass']
        self.redshift_gal = row['true_redshift_gal']

        self.ra = row['ra_gal']
        self.dec = row['dec_gal']

        self.gamma1 = row['gamma1']
        self.gamma2 = row['gamma2']

        self.median_major_axis = row['median_major_axis']
        self.scale_length = row['scale_length']
        self.bulge_fraction = row['bulge_fraction']

        self.disk_scalelength = row['disk_scalelength']
        self.disk_nsersic = row['disk_nsersic']
        self.disk_r50 = row['disk_r50']
        self.disk_ellipticity = row['disk_ellipticity']
       # self.disk_axis_ratio = row['disk_axis_ratio']

        self.bulge_r50 = row['bulge_r50']
        self.bulge_nsersic = row['bulge_nsersic']
        self.inclination_angle = row['inclination_angle']
        self.bulge_ellipticity = row['bulge_ellipticity']
      #  self.bulge_axis_ratio = row['bulge_axis_ratio']

        self.g_halo_flux_factor = gh_flux_factor
        self.halo_n = halo_n

        self.folding_threshold = folding_threshold
        self.maximum_fft_size = maximum_fft_size

        self.tot_flux = self.euclid_vis *10**(27)




    def make_galsim_galaxy(self, scale_up=1):

        gsparams = galsim.GSParams(
            folding_threshold=self.folding_threshold,
            maximum_fft_size=self.maximum_fft_size
        )

        def safe_e(e):
            if e is None:
                return 0.0
            e = float(e)
            if not np.isfinite(e):
                return 0.0
            return np.clip(e, 0.0, 0.8)   # GalSim becomes unstable above 0.9

        bulge_e = safe_e(self.bulge_ellipticity)
        disk_e  = safe_e(self.disk_ellipticity)

        g1 = float(self.gamma1)
        g2 = float(self.gamma2)

        bulge = galsim.Sersic(
            n=max(0.5, float(self.bulge_nsersic)),
            half_light_radius=max(0.001, scale_up * float(self.bulge_r50)),
            gsparams=gsparams
        )

        bulge = bulge.shear(e=bulge_e, beta=float(self.inclination_angle) * galsim.degrees)
        bulge_flux = self.tot_flux * float(self.bulge_fraction)
        bulge = bulge.withFlux(bulge_flux)

        gal = bulge


        if float(self.disk_r50) > 0:

            disk = galsim.Sersic(
                n=max(0.5, float(self.disk_nsersic)),
                half_light_radius=max(0.001, scale_up * float(self.disk_r50)),
                gsparams=gsparams
            )

            disk = disk.shear(e=disk_e, beta=float(self.disk_angle) * galsim.degrees)

            disk_flux = self.tot_flux * (1.0 - float(self.bulge_fraction))
            disk = disk.withFlux(disk_flux)

            gal += disk

        gal = gal.shear(g1=g1, g2=g2)

        return gal


    def postage_stamp_extent(self, stamp_size=64, scale=1):

        scale_deg = scale / 3600
        half_size_deg = (stamp_size / 2) * scale_deg

        ra_min = self.ra - half_size_deg
        ra_max = self.ra + half_size_deg
        dec_min = self.dec - half_size_deg
        dec_max = self.dec + half_size_deg

        return [ra_min, ra_max, dec_min, dec_max]

    def to_pixel(self, wcs):
        x_pix, y_pix = wcs.all_world2pix(self.ra, self.dec, 0)
        return x_pix, y_pix


    def adaptive_stamp_size(self, pixel_scale, fraction=0.99, min_pixels=64, growth_factor=20):
        R_bulge = self.sersic_radius_for_fraction(self.bulge_nsersic, self.bulge_r50, fraction) if self.bulge_r50>0 else 0
        R_disk  = self.sersic_radius_for_fraction(self.disk_nsersic, self.disk_r50, fraction) if self.disk_r50>0 else 0

        R_eff = self.bulge_fraction * R_bulge + (1 - self.bulge_fraction) * R_disk

        size_pix = min_pixels + growth_factor * (R_eff / pixel_scale)
        size_pix = int(np.ceil(size_pix / 2) * 2)  # force even number for FFT
        return size_pix


    def effective_hlr(self):
              f_b = self.bulge_fraction
              bulge = self.bulge_r50
              disk = self.disk_r50
              return f_b * bulge + (1 - f_b) * disk


    @staticmethod
    def sersic_radius_for_fraction(n, r_half, fraction=0.99):
        b_n = 2 * n - 0.324  # approximate
        x = gammaincinv(2*n, fraction)
        R = r_half * (x / b_n)**n
        return R



    def make_galsim_halo(self,halo_r50=None,ellipticity=0,gsparams=None,scale_up=1):

        if gsparams is None:
            gsparams = galsim.GSParams(
            self.folding_threshold,
            self.maximum_fft_size)

        if halo_r50 is None:
            if self.disk_r50 > 0:
                halo_r50 = 3.0 * float(self.disk_r50)
            else:
                halo_r50 = 3.0 * float(self.bulge_r50)



        total_flux = self.tot_flux
        halo_flux = total_flux * self.g_halo_flux_factor

        halo = galsim.Sersic(
            n=self.halo_n,
            half_light_radius=scale_up * halo_r50,
            gsparams=gsparams
        ).withFlux(halo_flux)

        if ellipticity > 0:
            halo = halo.shear(e=ellipticity, beta=0 * galsim.degrees)

        halo = halo.shear(g1=float(self.gamma1), g2=float(self.gamma2))

        return halo


    def adaptive_halo_stamp_size(self, pixel_scale, flux_fraction=0.9999, halo_r50=None, halo_n=1.0, scale_up=1, buffer_factor=1.2, max_pixels=4096):

        if halo_r50 is None:
            if self.disk_r50 > 0:
                halo_r50 = 3.0 * float(self.disk_r50)
            else:
                halo_r50 = 3.0 * float(self.bulge_r50)

        r_max = Galaxy.sersic_radius_for_fraction(halo_n, scale_up * halo_r50, flux_fraction)
        r_max *= buffer_factor
        stamp_size = int(np.ceil(2 * r_max / pixel_scale))

        if stamp_size % 2 == 0:
            stamp_size += 1

        stamp_size = min(stamp_size, max_pixels)

        return stamp_size
