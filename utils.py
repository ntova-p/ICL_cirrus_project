# -*- coding: utf-8 -*-

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u




def add_stamp_to_image(image, stamp, ra, dec, w):               #(image, stamp, g, w):

    #x_center, y_center = g.to_pixel(w)

    x_center, y_center = to_pixel_general(w,ra,dec)

    stamp_h, stamp_w = stamp.array.shape

    ix1 = int(x_center - stamp_w // 2)
    ix2 = ix1 + stamp_w
    iy1 = int(y_center - stamp_h // 2)
    iy2 = iy1 + stamp_h

    x1_img = max(ix1, 0)
    x2_img = min(ix2, image.array.shape[1])
    y1_img = max(iy1, 0)
    y2_img = min(iy2, image.array.shape[0])

    x1_stamp = x1_img - ix1
    x2_stamp = x2_img - ix1
    y1_stamp = y1_img - iy1
    y2_stamp = y2_img - iy1

    if (x2_img > x1_img) and (y2_img > y1_img):
        image.array[y1_img:y2_img, x1_img:x2_img] += stamp.array[y1_stamp:y2_stamp, x1_stamp:x2_stamp]

    return image


def to_pixel_general(wcs, ra,dec):
    x_pix, y_pix = wcs.all_world2pix(ra, dec, 0)
    return x_pix, y_pix

"""
def add_stamp_to_image(image, stamp, g, w, ra=None, dec=None):
    
    if ra is not None and dec is not None:
        x_center, y_center = w.wcs_world2pix([[ra, dec]], 1)[0]
    else:
        x_center, y_center = g.to_pixel(w)


    stamp_h, stamp_w = stamp.array.shape

    ix1 = int(x_center - stamp_w // 2)
    ix2 = ix1 + stamp_w
    iy1 = int(y_center - stamp_h // 2)
    iy2 = iy1 + stamp_h

    x1_img = max(ix1, 0)
    x2_img = min(ix2, image.array.shape[1])
    y1_img = max(iy1, 0)
    y2_img = min(iy2, image.array.shape[0])

    x1_stamp = x1_img - ix1
    x2_stamp = x2_img - ix1
    y1_stamp = y1_img - iy1
    y2_stamp = iy2 - iy1

    if (x2_img > x1_img) and (y2_img > y1_img):
        image.array[y1_img:y2_img, x1_img:x2_img] += stamp.array[y1_stamp:y2_stamp, x1_stamp:x2_stamp]

    return image

"""

def weighted_distribution_ellipse(points, weights, nsigma=1.5):
    w = np.asarray(weights)
    w /= w.sum()

    mu = np.average(points, axis=0, weights=w)

    x = points - mu
    cov = np.cov(x.T, aweights=w, bias=True)

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    axes_raw = np.sqrt(evals)  # semi-axes (unscaled)
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])  # major axis angle

    a = nsigma * axes_raw[0]  # scaled semi-major
    b = nsigma * axes_raw[1]  # scaled semi-minor

    ellipticity = 1.0 - b / a
    position_angle_deg = np.degrees(np.pi/2 - angle) % 360

    return {
        "center": mu,
        "a": a,
        "b": b,
        "ellipticity": ellipticity,
        "position_angle_deg": position_angle_deg,
        "axes": axes_raw,
        "angle": angle
    }


def populate_canvas_randomly(stamp_library, canvas_size=(3300, 3300), seed=None):

    
    if seed is not None:
        np.random.seed(seed)
    
    H, W = canvas_size
    canvas = np.zeros((H, W))
    canvas += 1e-50
    positions = []
    
    for stamp in stamp_library:
        h, w = stamp.shape

        y0 = np.random.randint(-h + 1, H)
        x0 = np.random.randint(-w + 1, W)
        
        y_start = max(y0, 0)
        y_end = min(y0 + h, H)
        x_start = max(x0, 0)
        x_end = min(x0 + w, W)
        
        stamp_y_start = max(0, -y0)
        stamp_y_end = stamp_y_start + (y_end - y_start)
        stamp_x_start = max(0, -x0)
        stamp_x_end = stamp_x_start + (x_end - x_start)
        
        canvas[y_start:y_end, x_start:x_end] += stamp[stamp_y_start:stamp_y_end, stamp_x_start:stamp_x_end]
        positions.append((y0, x0, h, w, y_start, y_end, x_start, x_end))
    
    return canvas, positions


def randomize_theta_dataframe(ra_array, dec_array, ra0, dec0, max_angle=30):
    
    center = SkyCoord(ra0*u.deg, dec0*u.deg, frame='icrs')
    points = SkyCoord(ra_array*u.deg, dec_array*u.deg, frame='icrs')
    
    r = center.separation(points)
    theta = center.position_angle(points)
    
    delta_theta = np.random.uniform(-max_angle, max_angle, size=len(ra_array)) * u.deg
    theta_new = theta + delta_theta
    
    new_points = center.directional_offset_by(theta_new, r)
    
    return new_points.ra.deg, new_points.dec.deg


def find_BCG(catalog):
    mean_ra = np.mean(catalog['ra_gal'])
    mean_dec = np.mean(catalog['dec_gal'])
    print(f'center = ra: {mean_ra} dec: {mean_dec}')

    dist = []

    for i in range(3):
        gal_ra = catalog['ra_gal'][i]
        gal_dec = catalog['dec_gal'][i]
        print(f'Galaxy {i} = ra: {gal_ra} dec: {gal_dec}')

        gal_dist = np.sqrt((mean_ra-gal_ra)**2+(mean_dec-gal_dec)**2)
        dist.append(gal_dist)

    print(f'distances = {dist}')
    bcg_idx = np.argmin(dist)
    print(f'the BCG is Galaxy {bcg_idx}')


    return(bcg_idx)