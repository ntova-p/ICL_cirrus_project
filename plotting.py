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
from astropy.stats import sigma_clipped_stats



def plot_epochs(items, title=None, save=False):

    ref = items[0]
    n = len(items)
    print("Halo:", ref.halo_id)
    print([obj.icl_re for obj in items])

    fig = plt.figure(figsize=(5*(n+2), 7))

    gs = GridSpec(
        3, n + 2,
        width_ratios=[1]*n + [1, 1.8],
        height_ratios=[0.35, 1, 1],
        figure=fig
    )

    pred_axes = [fig.add_subplot(gs[1, i]) for i in range(n)]
    res_axes  = [fig.add_subplot(gs[2, i]) for i in range(n)]
    ax_tgt    = fig.add_subplot(gs[1, n])
    ax_inp    = fig.add_subplot(gs[2, n])
    ax_prof   = fig.add_subplot(gs[1:3, n+1])
    ax_res_prof = ax_prof.inset_axes([0, -0.45, 1, 0.35])

    vmin, vmax = 1e-30, 1e-27
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for ax, obj in zip(pred_axes, items):
        ax.imshow(obj.pr, norm=LogNorm(vmin, vmax), cmap='gray_r')
        ax.set_title(f"Prediction\n{obj.epochs} epochs")
        ax.axis("off")

    for ax, obj in zip(res_axes, items):
        ax.imshow( obj.res, cmap='RdBu_r', norm=mcolors.SymLogNorm( linthresh=0.1, linscale=1.0, vmin=-1e-29, vmax=1e-29 ) )
        ax.set_title("Residual")
        ax.axis("off")

    ax_tgt.imshow(ref.tg, norm=LogNorm(vmin, vmax), cmap='gray_r')
    ax_tgt.set_title("Target")
    ax_tgt.axis("off")

    ax_inp.imshow(ref.inp, norm=LogNorm(vmin, vmax), cmap='gray_r')
    ax_inp.set_title("Input")
    ax_inp.axis("off")

    pixel_scale = 2
    z = ref.redshift

    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec).value
    kpc_per_pixel = kpc_per_arcsec * pixel_scale

    pixels_to_kpc = lambda x: x * kpc_per_pixel
    kpc_to_pixels = lambda x: x / kpc_per_pixel

    px_to_re = lambda x: x / ref.icl_re
    re_to_px = lambda x: x * ref.icl_re

    ymin, ymax = np.inf, -np.inf

    for c, obj in zip(colors, items):

        r = obj.pr_rp.radius
        mask = r >= obj.bcg_radius

        r_plot = r[mask]
        p_plot = obj.pr_rp.profile[mask]
        t_plot = np.interp( r_plot, ref.tg_rp.radius, ref.tg_rp.profile, left=np.nan, right=np.nan )

        res = p_plot - t_plot

        ax_prof.plot( r_plot, p_plot, color=c, label=f"{obj.epochs} epochs" )
        ax_res_prof.plot( r_plot, res, color=c )

        good = np.isfinite(p_plot) & (p_plot > 0)
        if np.any(good):
            ymin = min(ymin, np.nanmin(p_plot[good]))
            ymax = max(ymax, np.nanmax(p_plot[good]))

    r = ref.tg_rp.radius
    mask = r >= ref.bcg_radius

    r_plot = r[mask]
    t_plot = ref.tg_rp.profile[mask]
    ax_prof.plot( r_plot, t_plot, color="k", ls="--", lw=2, label="Target" )
    good = np.isfinite(t_plot) & (t_plot > 0)

    if np.any(good):
        ymin = min(ymin, np.nanmin(t_plot[good]))
        ymax = max(ymax, np.nanmax(t_plot[good]))

    ax_prof.axhline( mad_std(ref.inp), color="k", alpha=0.5, label="Input SNR" )
    ax_prof.set_yscale("log")
    # ax_prof.set_xlim(ref.bcg_radius, 128)

    if np.isfinite(ymin) and np.isfinite(ymax):
        ax_prof.set_ylim(ymin * 0.8, ymax * 1.2)

    xmin = ref.bcg_radius + 2
    xmax = 128

    ax_prof.set_xlim(xmin,xmax)
    
    #xticks = ax_prof.get_xticks()
    xticks = np.arange(np.ceil(xmin/10)*10, xmax+1, 20)
    ax_prof.set_xticks(xticks)
    ax_prof.set_xticklabels([f"{x*kpc_per_pixel:.1f}" for x in xticks])
    ax_prof.set_xlabel("Radius [kpc]")

    ax_re = ax_prof.secondary_xaxis("top", functions=(px_to_re, re_to_px))
    ax_re.set_xlabel(r"Radius [$r/r_e$]")

    ax_prof.set_ylabel("Surface Brightness")
    ax_prof.legend()


    ax_res_prof.axhline(0, color="k", lw=1)
    ax_res_prof.set_xlim(xmin,xmax)
    ax_res_prof.set_yscale("symlog", linthresh=1e-30)
    ax_res_prof.set_xlabel("Radius [pixels]")
    ax_res_prof.set_ylabel("Residual")
    ax_res_prof.grid(alpha=0.3)


    cbar1_ax = fig.add_axes([0.15, 0.02, 0.20, 0.02])
    cbar1 = fig.colorbar( pred_axes[0].images[0], cax=cbar1_ax, orientation="horizontal", )
    cbar1.set_label("Flux")

    cbar2_ax = fig.add_axes([0.45, 0.02, 0.20, 0.02])
    cbar2 = fig.colorbar( res_axes[0].images[0], cax=cbar2_ax, orientation="horizontal", )
    cbar2.set_label("Residual")

    fig.suptitle(f"Comparison: {title}, thr={ref.bcg_threshold}, halo id: {ref.halo_id} -- mass={ref.lm_halo:.1f}, z={ref.redshift:.2f}, doublet: {ref.doub}\n", fontsize=18)

    if save:
        plt.savefig(f'plots/{title} -- {ref.halo_id} -- {ref.idx}', bbox_inches='tight')
    
    return fig





def plot_thres(items, title=None):

    ref = items[0]
    n = len(items)

    

    fig = plt.figure(figsize=(5*(n+2), 8))

    gs = GridSpec( 3, n + 2, width_ratios=[1]*n + [1, 1.8], height_ratios=[0.35, 1, 1], figure=fig )

    pred_axes = [fig.add_subplot(gs[1, i]) for i in range(n)]
    res_axes  = [fig.add_subplot(gs[2, i]) for i in range(n)]
    ax_tgt    = fig.add_subplot(gs[1, n])
    ax_inp    = fig.add_subplot(gs[2, n])
    ax_prof   = fig.add_subplot(gs[1:3, n+1])
    ax_res_prof = ax_prof.inset_axes([0, -0.45, 1, 0.35])

    ymin,ymax=np.inf, -np.inf
    vmin, vmax = 1e-30, 1e-27
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for ax, obj in zip(pred_axes, items):
        ax.imshow(obj.pr, norm=LogNorm(vmin, vmax), cmap='gray_r')

        if obj.include_bcg:
            ax.set_title(f"Prediction\n no BCG mask")
        else:
            ax.set_title(f'Prediction\n threshold = {obj.bcg_threshold} ICL')

        ax.axis("off")

    for ax, obj in zip(res_axes, items):
        ax.imshow( obj.res, cmap='RdBu_r', norm=mcolors.SymLogNorm( linthresh=0.1, linscale=1.0, vmin=-1e-29, vmax=1e-29 ) )
        ax.set_title("Residual")
        ax.axis("off")

    ax_tgt.imshow(ref.tg, norm=LogNorm(vmin, vmax), cmap='gray_r')
    ax_tgt.set_title("Target")
    ax_tgt.axis("off")

    ax_inp.imshow(ref.inp, norm=LogNorm(vmin, vmax), cmap='gray_r')
    ax_inp.set_title("Input")
    ax_inp.axis("off")





    pixel_scale = 2
    z = ref.redshift

    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec).value
    kpc_per_pixel = kpc_per_arcsec * pixel_scale
    px_to_re = lambda x: x / ref.icl_re
    re_to_px = lambda x: x * ref.icl_re

    obj = items[0]
    r = obj.tg_rp.radius
    mask = r >= obj.bcg_radius
    r_plot = r[mask]
    p_plot = obj.pr_rp.profile[mask]
    t_plot = obj.tg_rp.profile[mask]
    res = p_plot - t_plot

    ax_prof.plot( r_plot, p_plot, color='C0', lw=2, label=f"Prediction ({obj.epochs})" )
    ax_res_prof.plot( r_plot, res, color='C0', lw=2 )

    good = np.isfinite(p_plot) & (p_plot > 0)
    if np.any(good):
        ymin = min(ymin, np.nanmin(p_plot[good]))
        ymax = max(ymax, np.nanmax(p_plot[good]))


    ax_prof.plot( r_plot, t_plot, color='k', ls='--', lw=2, label='Target' )

    good = np.isfinite(t_plot) & (t_plot > 0)
    if np.any(good):
        ymin = min(ymin, np.nanmin(t_plot[good]))
        ymax = max(ymax, np.nanmax(t_plot[good]))


    ax_prof.axhline( mad_std(ref.inp), color='k', alpha=0.5, label='Input SNR' )



    if np.isfinite(ymin) and np.isfinite(ymax):
        ax_prof.set_ylim(ymin*0.8, ymax*1.2)

    ax_prof.set_xlim(0,128)

    for c, obj in zip(colors[1:], items[1:]):
        ax_prof.axvline( obj.bcg_radius, color='k', ls=':', lw=2, alpha=0.9, label=f'thr={obj.bcg_threshold:.1}')
        ax_res_prof.axvline( obj.bcg_radius, color='k', ls=':', lw=2, alpha=0.9 )


    ax_prof.set_yscale("log")
    ax_prof.set_ylim(ymin*0.8, ymax*1.2)

    xticks = ax_prof.get_xticks()
    ax_prof.set_xticks(xticks)
    ax_prof.set_xticklabels([f"{x*kpc_per_pixel:.1f}" for x in xticks])

    ax_prof.set_xlabel("Radius [kpc]")
    ax_prof.set_ylabel("Surface Brightness")

    ax_re = ax_prof.secondary_xaxis( "top", functions=(px_to_re, re_to_px) )
    ax_re.set_xlabel(r"Radius [$r/r_e$]")

    ax_prof.legend()

    ax_res_prof.set_xlim(0,128)
    ax_res_prof.axhline(0, color='k', lw=1)
    ax_res_prof.set_yscale("symlog", linthresh=1e-30)
    ax_res_prof.set_xlabel("Radius [pixels]")
    ax_res_prof.set_ylabel("Residual")
    ax_res_prof.grid(alpha=0.3)


    cbar1_ax = fig.add_axes([0.15, 0.02, 0.20, 0.02])
    cbar1 = fig.colorbar( pred_axes[0].images[0], cax=cbar1_ax, orientation="horizontal", )
    cbar1.set_label("Flux")

    cbar2_ax = fig.add_axes([0.45, 0.02, 0.20, 0.02])
    cbar2 = fig.colorbar( res_axes[0].images[0], cax=cbar2_ax, orientation="horizontal", )
    cbar2.set_label("Residual")


    fig.suptitle(
        f"Comparison: {title}, halo id: {ref.halo_id} "
        f"-- mass={ref.lm_halo:.1f}, z={ref.redshift:.2f}, "
        f"doublet: {ref.doub}",
        fontsize=18,
    )

    fig.subplots_adjust( left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25, hspace=0.25, )

    plt.savefig(f"plots/{title} -- {ref.halo_id} -- {obj.idx}", bbox_inches='tight')

    return fig