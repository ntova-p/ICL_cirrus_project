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



info = fits.open('../clusters_metadata.fits')

def get_hdu_obj(hdul, idx, hdu_type):
    return next(
        hdu for hdu in hdul
        if hdu.header.get("INDEX") == idx
        and hdu.header.get("TYPE") == hdu_type
    )

halo_table = info[1].data
halo_lookup = {row['halo_id']: row
               for row in halo_table}




class item:

    def __init__(self, hdul, epochs, type, idx, halo_lookup=None, include_bcg=True, component_path="/data2/ntova/new_component_fits"):

        self.idx=idx
        self.epochs = epochs
        self.type = type
        self.include_bcg = include_bcg
        self.component_path = component_path     

        self.input_hdu = get_hdu_obj(hdul, idx, "INPUT")  
        self.prediction_hdu = get_hdu_obj(hdul, idx, "PRED")
        self.target_hdu = get_hdu_obj(hdul, idx, "TARGET")

        self.header = self.input_hdu.header
        self.halo_id= self.header['HALO_ID']
        self.doub = self.header['DOUBLET']   
        
        self.inp = 10**self.input_hdu.data
        self.pr = 10**self.prediction_hdu.data
        self.tg = 10**self.target_hdu.data

        self.inp_raw = self.inp.copy()
        self.pr_raw  = self.pr.copy()
        self.tg_raw  = self.tg.copy()

        #self.make_bcg_mask()

        #if not self.include_bcg:

         #   self.inp *= self.analysis_mask
         #   self.pr  *= self.analysis_mask
         #   self.tg  *= self.analysis_mask

        self.res = self.pr - self.tg
        self.cir_res_real = self.inp - self.tg
        self.cir_res_est = self.inp - self.pr
        





        if halo_lookup is not None:
            self.halo_info = halo_lookup.get(self.halo_id)

            if self.halo_info is not None:
                self.lm_halo = self.halo_info['lm_halo']
                self.redshift = self.halo_info['true_redshift_halo']
                self.n_galaxies = self.halo_info['n_galaxies']
                self.ra_min = self.halo_info['ra_min']
                self.ra_max = self.halo_info['ra_max']
                self.dec_min = self.halo_info['dec_min']
                self.dec_max = self.halo_info['dec_max']

            




    def make_bcg_mask(self):
        if self.include_bcg:
            self.bcg_radius = 0
            self.analysis_mask = np.ones_like(self.tg, dtype=bool)
            return
        
        try:
            icl = fits.open( f"{self.component_path}/{self.halo_id}_ICL_4_0.fits" )[0].data
            icl_bcg = fits.open( f"{self.component_path}/{self.halo_id}_ICL_BCG.fits" )[0].data

        except FileNotFoundError:

            print(f"BCG files missing for halo {self.halo_id}")

            self.bcg_radius = np.nan
            self.analysis_mask = np.ones_like(self.tg, dtype=bool)

            return
        
        bcg = icl_bcg - icl
        cen = centroid_2dg(icl)
        edge_radii = np.arange(icl.shape[0] // 2)

        rp_icl = RadialProfile(icl, cen, edge_radii)
        rp_bcg = RadialProfile(bcg, cen, edge_radii)

        ratio = rp_bcg.profile / (rp_icl.profile + 1e-30)
        mask = ratio < 0.1

        if np.any(mask):
            r10 = rp_icl.radius[np.where(mask)[0][0]]
        else:
            r10 = 0

        
        self.bcg_radius = r10
        
        y, x = np.indices(self.tg.shape)
        cx, cy = cen
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        self.analysis_mask = r >= r10


    def find_centers(self):
        self.inpcen = centroid_2dg(self.inp)  
        self.prcen  = centroid_2dg(self.pr)
        self.tgcen  = centroid_2dg(self.tg)
        self.tgcen = np.asarray(self.tgcen).ravel()
        self.prcen = np.asarray(self.prcen).ravel()
        self.diffcen = np.hypot(self.prcen[0] - self.tgcen[0], self.prcen[1] - self.tgcen[1])
        
    def find_fluxes(self):
        self.pr_flux = np.sum(self.pr)
        self.tg_flux = np.sum(self.tg)
        self.diffflux = self.pr_flux - self.tg_flux
        self.flux_frac = 100 * self.diffflux / self.tg_flux   

    def radial_profiles(self):
        edge_radii = np.arange(len(self.pr) / 2)
        self.pr_rp  = RadialProfile(self.pr,  self.prcen,  edge_radii)
        self.tg_rp  = RadialProfile(self.tg,  self.tgcen,  edge_radii)
        self.inp_rp = RadialProfile(self.inp, self.inpcen, edge_radii)

        y, x = np.indices(self.pr.shape)
        cx, cy = self.prcen
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = r <= len(self.pr)/2
        self.pr_points = [r[mask], self.pr[mask]]


    def process(self):
        self.find_centers()
        self.find_fluxes()
        self.radial_profiles()

        

    def power_spectrum(self, image, pixscale=2.0, nbins=40):       ## apodisation

        image = np.asarray(image, float)
        ny, nx = image.shape

        F = np.fft.fft2(image)
        P2 = np.abs(F)**2

        ky = np.fft.fftfreq(ny, d=pixscale)
        kx = np.fft.fftfreq(nx, d=pixscale)
        KX, KY = np.meshgrid(kx, ky)
        kr = np.sqrt(KX**2 + KY**2)

        kmin = kr[kr > 0].min()
        kmax = kr.max()
        bins = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)

        kcen = 0.5 * (bins[:-1] + bins[1:])
        p1d = np.full(nbins, np.nan)

        for i in range(nbins):
            m = (kr >= bins[i]) & (kr < bins[i + 1])
            if np.any(m):
                p1d[i] = P2[m].mean()

        return kcen, p1d
    

    def ps_res(self):
        return self.power_spectrum(self.res)

    def ps_cir_real(self):
        return self.power_spectrum(self.cir_res_real)

    def ps_cir_est(self):
        return self.power_spectrum(self.cir_res_est)

    def ps_input(self):
        return self.power_spectrum(self.inp)

    def ps_pred(self):
        return self.power_spectrum(self.pr)
    
    def get_all_power_spectra(self, pixscale=1.0, nbins=40):

        spectra = {
            "Input": self.inp,
            "Prediction": self.pr,
            "Target": self.tg,
            "Residual (pr - tg)": self.res,
            "Cirrus real (inp - tg)": self.cir_res_real,
            "Cirrus est (inp - pr)": self.cir_res_est,
        }

        results = {}

        for name, img in spectra.items():
            k, p = self.power_spectrum(img, pixscale=pixscale, nbins=nbins)
            results[name] = (k, p)

        return results


####################################################### PLOTS ####################################################################
 

    def plot_power_spectra(self, pixscale=1.0, nbins=40):

        spectra = self.get_all_power_spectra(pixscale, nbins)
        plt.figure(figsize=(8, 6))

        for name, (k, p) in spectra.items():
            mask = np.isfinite(p)

            if name in ["Input", "Prediction", "Target", "Residual (pr - tg)"]:
                plt.loglog( k[mask], p[mask], linestyle=":", alpha=0.5, label=name )

            else:
                plt.loglog( k[mask], p[mask], linewidth=2, alpha=0.9, label=name )

        plt.xlabel("k [1/angle]")
        plt.ylabel("P(k)")
        plt.title(f"Power spectra — Halo {self.halo_id} - {self.type}, {self.epochs} epochs")
        plt.legend()
        #plt.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.show()


        
    def plot_images(self):
       
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))

        imgs = [self.inp, self.tg, self.pr]
        titles = [ "input", "target", "prediction"]
        vmin, vmax = 1e-30, 1e-26

        for i in range(3):
            im = ax[i].imshow(imgs[i], norm=LogNorm(vmin, vmax), cmap="gray_r")
            ax[i].set_title(titles[i])
            ax[i].axis("off")

        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]

        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("log10 Flux")

        plt.suptitle(f'Halo {self.halo_id} [idx = {self.idx}] -- {self.type}, {self.epochs} epochs', fontsize=18)
       # plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_radial_profiles(self):
        
        fig, ax = plt.subplots(figsize=(8, 6))

        self.pr_rp.plot(ax=ax, color="blue", label="Prediction")
        self.tg_rp.plot(ax=ax, color="red", ls="--", label="Target")
        self.inp_rp.plot(ax=ax, color="k", alpha=0.5, ls=":", label="Input")

        ax.axhline(y=mad_std(self.inp), color="k", label="Input SNR")

        ax.set_yscale("log")
        ax.set_xlabel("Radius [pixels]")
        ax.set_ylabel("Surface Brightness")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_item(self):
            
            print("tgcen:", self.tgcen, type(self.tgcen), np.shape(self.tgcen))
            print("prcen:", self.prcen, type(self.prcen), np.shape(self.prcen))
            print("diffcen:", self.diffcen)

            fig = plt.figure(figsize=(16, 8))
            #gs = GridSpec(2, 3, width_ratios=[1, 1, 1.8], figure=fig)
            gs = GridSpec(
                3, 3,
                width_ratios=[1, 1, 1.8],
                height_ratios=[0.35, 1, 1],
                figure=fig
            )
            ax_pred = fig.add_subplot(gs[1, 0])
            ax_tgt  = fig.add_subplot(gs[1, 1])
            ax_inp  = fig.add_subplot(gs[2, 0])
            ax_res  = fig.add_subplot(gs[2, 1])
            ax_prof = fig.add_subplot(gs[1:3, 2])
            ax_res_prof = ax_prof.inset_axes([0, -0.45, 1, 0.35])
            

            
            vmin, vmax = 1e-30, 1e-28
            cmap = 'viridis'

            im0 = ax_pred.imshow(self.pr,norm=LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
            ax_tgt.imshow(self.tg,norm=LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
            ax_inp.imshow(self.inp,norm=LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
            im_res = ax_res.imshow(self.res,cmap='RdBu_r', norm= mcolors.SymLogNorm(linthresh=0.1, linscale=1.0, vmin=-1e-29, vmax=1e-29)) #vmin=-1e-29,vmax=1e-29,

            ax_pred.set_title("Prediction")
            ax_tgt.set_title("Target")
            ax_inp.set_title("Input")
            ax_res.set_title("Residual")

            for ax in [ax_pred, ax_tgt, ax_inp, ax_res]:
                ax.axis("off")

            self.pr_rp.plot(ax=ax_prof,color='blue',label='prediction')
            self.tg_rp.plot(ax=ax_prof,color='red',ls='--',label='target')
            self.inp_rp.plot(ax=ax_prof,color='k',alpha=0.3,ls='dotted',label='input')

            ax_prof.axhline(y=mad_std(self.inp),label='input SNR',color='k')
            ax_prof.scatter(self.pr_points[0],self.pr_points[1],s=1,alpha=0.01,color='cyan')

            ax_prof.set_yscale('log')
            ax_prof.set_xlabel("Radius [pixels]")
            ax_prof.set_ylabel("Surface Brightness")

            ax_prof.legend()

            r = self.pr_rp.radius
            p = self.pr_rp.profile
            t = self.tg_rp.profile

            res = p - t

            ax_res_prof.axhline(0, color='k', lw=1)
            ax_res_prof.plot(r, res, color='blue', label='Pred - Target')

            ax_res_prof.set_yscale('symlog', linthresh=1e-30)
            ax_res_prof.set_xlabel("Radius [pixels]")
            ax_res_prof.set_ylabel("Residual")
            ax_res_prof.grid(alpha=0.3)

            ax_prof.set_xlabel("")
            plt.setp(ax_prof.get_xticklabels(), visible=False)



            txt_cent = (
                f"Target center: ({self.tgcen[0]:.2f}, {self.tgcen[1]:.2f})\n"
                f"Prediction center: ({self.prcen[0]:.2f}, {self.prcen[1]:.2f})\n"
                f"Center offset: {self.diffcen:.1f} px\n\n"
            )

            txt_flux = (
                f"Target flux: {self.tg_flux:.3e}\n"
                f"Prediction flux: {self.pr_flux:.3e}\n"
                f"Flux difference: {self.diffflux:.3e}\n"
                f"Fractional difference: {self.flux_frac:.2f}%"
            )

            fig.text(0.65,0.92,txt_cent,ha='left',va='top',fontsize=11)
            fig.text(0.98,0.92,txt_flux,ha='right',va='top',fontsize=11)
            fig.suptitle(f"{self.type} -- halo id: {self.halo_id}, doublet: {self.doub}\n", fontsize=18)

            cbar1_ax = fig.add_axes([0.02, 0.02, 0.2, 0.02])  # [left, bottom, width, height]
            cbar1 = fig.colorbar(im0, cax=cbar1_ax, orientation="horizontal")
            cbar1.set_label("log10 Flux")
            cbar2_ax = fig.add_axes([0.30, 0.02, 0.2, 0.02])
            cbar2 = fig.colorbar(im_res, cax=cbar2_ax, orientation="horizontal")
            cbar2.set_label("Residual")


            fig.subplots_adjust(
                                left=0.05,
                                right=0.98,
                                top=0.92,
                                bottom=0.08,
                                wspace=0.25,
                                hspace=0.25
                            )
            
            #plt.tight_layout(rect=[0, 0, 1, 0.92])

