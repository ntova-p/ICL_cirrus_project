# -*- coding: utf-8 -*-


import numpy as np
from astropy import wcs
import galsim
from scipy.special import gamma, gammaincinv



class ICL:

    def __init__(self, bcg, wcs, m200, cluster_flux, flux_fraction=0.1, pixel_scale=None, n_sersic=1.0, e=0.0, pa=0.0, r_half=None):

        self.ra = bcg.ra
        self.dec = bcg.dec
        self.bcg_loglum = bcg.log_lum
        self.bcg_euclid_vis = bcg.euclid_vis
        self.cluster_flux = cluster_flux    ## this is the total galaxy flux!!
        self.z = bcg.redshift_gal

        self.m200 = np.log10(m200)
        self.pixel_scale = pixel_scale

        self.wcs = wcs
        self.x, self.y = wcs.all_world2pix(self.ra, self.dec, 0)


        self.flux_fraction = 0.1569*(self.m200)-1.8758  # S.L. Ahad 2022
        self.n_sersic = n_sersic
        self.e = np.clip(e, 0.0, 0.8)
        self.pa = pa

        self.r_half = r_half
        self.flux = (self.flux_fraction * (self.cluster_flux))/(1-self.flux_fraction)
        self.profile = None

        print(f'Cluster flux: {self.flux} ---- ICL flux fraction: {self.flux_fraction}')

    @staticmethod
    def scale_radius_to_hlr(r_s, n):
        b_n = 2 * n - 0.324
        return r_s * (b_n ** n)


    @staticmethod
    def b_n(n):
        return 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2)


    def compute_re_from_mu0(self, mu0, ZP):
        n = self.n_sersic
        b_n = self.b_n(n)
        b_4 = self.b_n(4)
        F = self.flux
        print(F)

        I04 = 10**(-0.4 * (mu0 - ZP))
        IE4 = I04*np.exp(-b_n)
        h = np.sqrt(F/(np.pi*gamma(2*n+1)*I04*np.exp(-b_4+b_n)))

        return h


    def make_profile(self, mu0, ZP):

        self.r_s = self.compute_re_from_mu0(mu0, ZP)
        self.profile = (
            galsim.Sersic(n=self.n_sersic, scale_radius=self.r_s)
            .shear(e=self.e, beta=self.pa * galsim.degrees)
            .withFlux(self.flux*1e27)
        )

    def adaptive_stamp_size(self, pixel_scale, flux_fraction=0.999, buffer_factor=1.2, max_pixels=4096):

        if not hasattr(self, "r_s"):
            raise RuntimeError("Profile must be built before computing stamp size")

        b_n = self.b_n(self.n_sersic)
        r_half = self.r_s * b_n**self.n_sersic

        R_frac = r_half * (gammaincinv(2*self.n_sersic, flux_fraction) / b_n)**self.n_sersic
        R_max = R_frac * buffer_factor

        stamp_size = int(np.ceil(2 * R_max / pixel_scale))
        if stamp_size % 2 == 0:
            stamp_size += 1

        return min(stamp_size, max_pixels)


    def draw_stamp(self, pixel_scale, mu0=None, ZP=None, stamp_size=None):

        if self.profile is None:
            if mu0 is None or ZP is None:
                raise ValueError("Must provide mu0 and ZP to build profile")
            self.make_profile(mu0, ZP)

        if stamp_size is None:
            stamp_size = self.adaptive_stamp_size(pixel_scale)

        img = self.profile.drawImage(
            nx=stamp_size,
            ny=stamp_size,
            scale=pixel_scale,
            method="auto"
        )

        return img
