import numpy as np
import matplotlib.pyplot as plt
import photutils
from astropy.io import fits
from matplotlib.colors import LogNorm
from photutils.centroids import centroid_2dg
from photutils.profiles import RadialProfile
from astropy.stats import mad_std
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import copy
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.special import gamma
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit




info = fits.open('../clusters_metadata.fits')

from item_class import item, get_hdu_obj

halo_table = info[1].data
halo_lookup = {row['halo_id']: row
               for row in halo_table}



class trial:

    def __init__(self, hdul, type, epochs, halo_lookup=halo_lookup, bcg_threshold=0.1, include_bcg=True,  component_path="/data2/ntova/new_component_fits"):

        self.type = type
        self.epochs = epochs
        self.hdul=hdul
        self.include_bcg = include_bcg
        self.component_path = component_path     
        self.bcg_threshold= bcg_threshold
        

        self.items = []

        self.diffcen = None
        self.diffflux = None
        self.chisq = None


    def collect(self):

        def arr(attr):
            return np.array([getattr(obj, attr, np.nan) for obj in self.items])

        self.diffcen = arr("diffcen")
        self.diffflux = arr("diffflux")
        self.flux_frac = arr("flux_frac")

        self.lm_halo = arr("lm_halo")
        self.redshift = arr("redshift")
        self.n_galaxies = arr("n_galaxies")
        self.halo_id = arr("halo_id")


    def process_all_hdus(self):
        self.items = []
        n_items = int(len(self.hdul) / 3) - 1

        for idx in range(n_items):
            obj = item(self.hdul, epochs=self.epochs, type=self.type, idx=idx, halo_lookup=halo_lookup, include_bcg=self.include_bcg, bcg_threshold=self.bcg_threshold, component_path=self.component_path)
            obj.process()
            self.items.append(obj)   
            
        self.collect()


    
    def select(self, mask):
        new = copy.copy(self)
        new.items = [
            obj for obj, keep in zip(self.items, mask)
            if keep]
        new.collect()

        return new
    
    def high_mass(self, threshold=None):
        if threshold is None:
            threshold = np.nanmedian(self.lm_halo)
            self.mass_threshold=threshold

        return self.select(self.lm_halo >= threshold)


    def low_mass(self, threshold=None):
        if threshold is None:
            threshold = np.nanmedian(self.lm_halo)

        return self.select(self.lm_halo < threshold)


    def high_redshift(self, threshold=None):
        if threshold is None:
            threshold = np.nanmedian(self.redshift)
            self.z_threshold=threshold

        return self.select(self.redshift >= threshold)


    def low_redshift(self, threshold=None):
        if threshold is None:
            threshold = np.nanmedian(self.redshift)

        return self.select(self.redshift < threshold)



    def summary(self):
        return {
            "N": len(self.items),
            "diffcen_mean": np.mean(self.diffcen),
            "diffcen_median": np.median(self.diffcen),
            "diffcen_std": np.std(self.diffcen),

            "diffflux_mean": np.mean(self.diffflux),
            "diffflux_median": np.median(self.diffflux),
            "diffflux_std": np.std(self.diffflux),
        }
    

    
################################################### PLOTTING


    def plot_all_power_spectra(self):

        if len(self.items) == 0:
            return

        k, p = self.items[0].spectra["Cirrus real (inp - tg)"]

        sum_real = np.zeros_like(p)
        sum_pred = np.zeros_like(p)

        for obj in self.items:

            k_real, p_real = obj.spectra["Cirrus real (inp - tg)"]
            k_pred, p_pred = obj.spectra["Cirrus est (inp - pr)"]

            plt.loglog(k_real, p_real, color="green", alpha=0.05)
            plt.loglog(k_pred, p_pred, color="pink", alpha=0.05)

            sum_real += p_real
            sum_pred += p_pred

        mean_real = sum_real / len(self.items)
        mean_pred = sum_pred / len(self.items)

        plt.loglog(k, mean_real, color="green", lw=3, label="Mean cirrus (real)")
        plt.loglog(k, mean_pred, color="pink", lw=3, label="Mean cirrus (estimated)")

        plt.xlabel("k [1/angle]")
        plt.ylabel("P(k)")
        plt.title(f"Power spectra for {self.type}")
        plt.legend()
        plt.show()


    def plot_all_power_spectra_OLD(self):

        k0, p0 = self.items[0].spectra["Cirrus real (inp - tg)"]
        sum_real = np.zeros_like(p0)
        sum_pred = np.zeros_like(p0)
        
        n = len(self.items)

        for i in self.items:
            
            for name, (k, p) in i.spectra.items():
                mask = np.isfinite(p)

                if name in ["Cirrus real (inp - tg)"]:
                    plt.loglog( k[mask], p[mask], linewidth=2, alpha=0.05, color='green')
                    sum_real += p
                
                if name in ["Cirrus est (inp - pr)"]:
                    plt.loglog( k[mask], p[mask], linewidth=2, alpha=0.05, color='pink')
                    sum_pred += p
            
            

        plt.loglog(k0, sum_real/n,  label='Mean Target', linestyle='--', color='green', lw=2)
        plt.loglog(k0, sum_pred/n, label='Mean Prediction', linestyle='--', color='pink', lw=2)
        plt.xlabel('k [1/angle]')
        plt.ylabel('P(k)')
        plt.legend()
        plt.title(f'Power Spectra for {self.type}')

        plt.show()






    def plot_split_hist(
        self,
        param="flux_frac",
        mass_thr=14.8,
        z_thr=0.35,
        bins=15,
        xlabel=None,
        title_mass="Mass split",
        title_z="Redshift split"
    ):

        values = getattr(self, param)

        hist_total, bin_edges = np.histogram(values, bins=bins)
        x_min, x_max = bin_edges[0], bin_edges[-1]
        y_max = hist_total.max()

        low_mass  = self.lm_halo < mass_thr
        high_mass = self.lm_halo >= mass_thr

        low_z  = self.redshift < z_thr
        high_z = self.redshift >= z_thr



        lm = self.select(low_mass)
        hm = self.select(high_mass)
        lz = self.select(low_z)
        hz = self.select(high_z)

        lm_v = getattr(lm, param)
        hm_v = getattr(hm, param)
        lz_v = getattr(lz, param)
        hz_v = getattr(hz, param)

        bins = np.linspace( np.nanmin(values), np.nanmax(values), bins )

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1.8, 1, 1])

        ax_total = fig.add_subplot(gs[:, 0])
        ax_lm = fig.add_subplot(gs[0, 1])
        ax_hm = fig.add_subplot(gs[0, 2])
        ax_lz = fig.add_subplot(gs[1, 1])
        ax_hz = fig.add_subplot(gs[1, 2])

        if xlabel is None:
            xlabel = param

        ax_total.hist(values, bins=bins, color="0.4", alpha=0.6, edgecolor="black")
        ax_total.set_title("All Halos")
        ax_total.set_xlabel(xlabel)
        ax_total.set_ylabel("Count")
        ax_total.axvline(np.mean(values), color = color, ls='--', label='Mean')

        panels = [
            (ax_lm, lm_v, f"Low Mass (M<{mass_thr})", "tab:green"),
            (ax_hm, hm_v, f"High Mass (M>{mass_thr})", "tab:orange"),
            (ax_lz, lz_v, f"Low z (z<{z_thr})", "tab:blue"),
            (ax_hz, hz_v, f"High z (z>{z_thr})", "tab:red"),
        ]

        for ax, data, title, color in panels:

            ax.hist( data, bins=bins, color=color, alpha=0.5, edgecolor=color, linewidth=1)
        #    ax.axvline(np.nanmedian(data), color="black", ls="--", lw=1)
            ax.axvline(np.mean(data), color = color, ls='--', label='Mean')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")

        axes = [ax_total, ax_lm, ax_hm, ax_lz, ax_hz]
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max+2)

        if self.include_bcg=='True':
            fig.suptitle( f"{self.type} | BCG included | epochs={self.epochs} | param={param}", fontsize=14 )
        else:
            fig.suptitle( f"{self.type} | BCG excluded (for BCG>{self.bcg_threshold}*ICL) | epochs={self.epochs} | param={param}", fontsize=14 )


        plt.tight_layout()
        #plt.show()
        #plt.savefig(f"{self.type} | epochs={self.epochs} | param={param}")

        return fig
