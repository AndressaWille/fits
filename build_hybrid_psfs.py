#!/usr/bin/env python3
"""Build hybrid PSFs with an empirical core and STPSF/WebbPSF model wings.

This empirical-to-simulated hybrid approach was used in Sarrouh et al. (2025).
The empirical PSF has unit weight inside ``HYBRID_R1_ARCSEC``. Its weight
decreases linearly to zero at ``HYBRID_R2_ARCSEC``; only the simulated model
is used at larger radii.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.visualization import simple_norm
from photutils.centroids import centroid_quadratic
from scipy.ndimage import shift, zoom


DATA_DIR = Path(__file__).resolve().parent
SIM_PSF_DIR = Path("/home/andressa/Doutorado/Pesquisa/sim_psfs")
FILTERS = [
    "f090w", "f105w", "f110w", "f115w", "f125w", "f140w", "f150w",
    "f200w", "f250m", "f300m", "f410m", "f444w",
]
REGION_FILE_OVERRIDES = {"f300m": "stars_f300m_new_2.reg"}
WEIGHTED_SIGMA_CLIPPED_FILTERS = set()

CUTOUT_SIZE = 151
APERTURE_RADIUS = 30
BACKGROUND_BORDER = 15
SIGMA_CLIP = 2
SAVED_PSF_SIZE = 101
OVERSAMPLING = 2

# The active values can be adjusted after inspecting the diagnostic figures.
HYBRID_R1_ARCSEC = 0.5
HYBRID_R2_ARCSEC = 0.75

# Save diagnostic plots to ``figuras_psf/<filter>_*.png``.
SAVE_FIGURES = True
SHOW_FIGURES = False
FIGURE_DIR = DATA_DIR / "figuras_psf"


def read_ds9_boxes(region_file):
    """Read star names and pixel coordinates from DS9 box regions."""
    stars = []
    with open(region_file, encoding="utf-8") as file:
        for line in file:
            if not line.startswith("box"):
                continue
            match = re.search(
                r"box\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\).*text=\{([^}]+)\}", line
            )
            if match is not None:
                x, y, _width, _height, _angle, name = match.groups()
                stars.append({"name": name, "x": float(x), "y": float(y)})
    return pd.DataFrame(stars)


def extract_cutouts(image, stars, size):
    """Extract square cutouts centred on the selected stars."""
    return {
        star.name: Cutout2D(
            image, (star.x, star.y), size=size, mode="partial", fill_value=np.nan
        )
        for _, star in stars.iterrows()
    }


def subtract_background(cutout, border, sigma=3):
    """Subtract a sigma-clipped scalar background measured at the cutout edges."""
    image = np.asarray(cutout, dtype=float).copy()
    edge_mask = np.zeros_like(image, dtype=bool)
    edge_mask[:border, :] = edge_mask[-border:, :] = True
    edge_mask[:, :border] = edge_mask[:, -border:] = True
    _, background, stddev = sigma_clipped_stats(image[edge_mask], sigma=sigma)
    return image - background, background, stddev


def measure_centers(cutouts):
    """Measure quadratic centroids for a dictionary of two-dimensional images."""
    centers = {}
    for name, image in cutouts.items():
        x_center, y_center = centroid_quadratic(np.asarray(image))
        centers[name] = {"x": x_center, "y": y_center}
    return centers


def recenter_cutouts(cutouts, centers, order=3):
    """Shift each cutout so its measured centroid matches its geometric centre."""
    recentered = {}
    for name, image in cutouts.items():
        image = np.asarray(image)
        ny, nx = image.shape
        dx = (nx - 1) / 2 - centers[name]["x"]
        dy = (ny - 1) / 2 - centers[name]["y"]
        recentered[name] = shift(
            image, (dy, dx), order=order, mode="constant", cval=np.nan, prefilter=True
        )
    return recentered


def normalize_by_aperture(cutouts, radius):
    """Normalize each image by its flux within a central circular aperture."""
    normalized, fluxes = {}, {}
    for name, image in cutouts.items():
        image = np.asarray(image)
        y, x = np.indices(image.shape)
        radius_map = np.hypot(x - (image.shape[1] - 1) / 2, y - (image.shape[0] - 1) / 2)
        flux = np.nansum(image[radius_map <= radius])
        normalized[name] = image / flux
        fluxes[name] = flux
    return normalized, fluxes


def weighted_sigma_clipped_psf(cutouts, weights, sigma):
    """Build a weighted empirical PSF while rejecting pixel-wise outliers."""
    names = list(cutouts)
    images = np.array([cutouts[name] for name in names])
    weight_array = np.array([weights[name] for name in names])
    clipped = sigma_clip(images, sigma=sigma, axis=0)
    numerator = np.nansum(clipped * weight_array[:, None, None], axis=0)
    denominator = np.nansum((~clipped.mask) * weight_array[:, None, None], axis=0)
    return numerator / denominator


def radial_profile(image):
    """Return the mean flux in successive one-pixel radial annuli."""
    y, x = np.indices(image.shape)
    radius = np.hypot(x - (image.shape[1] - 1) / 2, y - (image.shape[0] - 1) / 2)
    radius_int = radius.astype(int)
    profile = np.array([np.nanmean(image[radius_int == value]) for value in range(radius_int.max())])
    return np.arange(len(profile)), profile


def encircled_energy(image):
    """Return cumulative normalized flux as a function of pixel radius."""
    y, x = np.indices(image.shape)
    radius = np.hypot(x - (image.shape[1] - 1) / 2, y - (image.shape[0] - 1) / 2).ravel()
    flux = image.ravel()
    valid = np.isfinite(flux)
    order = np.argsort(radius[valid])
    radius, flux = radius[valid][order], flux[valid][order]
    return radius, np.cumsum(flux) / np.nansum(flux)


def normalize_psf(psf):
    """Replace invalid values, clip negatives, and normalize the total flux to one."""
    psf = np.array(psf, dtype=float, copy=True)
    psf[~np.isfinite(psf)] = 0.0
    psf[psf < 0] = 0.0
    total = psf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"The PSF cannot be normalized; total={total}.")
    return psf / total


def center_crop(image, size):
    """Extract a square crop centred on an image."""
    ny, nx = image.shape
    if size > ny or size > nx:
        raise ValueError(f"Cannot extract a {size}x{size} crop from a {nx}x{ny} PSF.")
    y0, x0 = (ny - size) // 2, (nx - size) // 2
    return np.array(image[y0:y0 + size, x0:x0 + size], copy=True)


def center_match_shape(image, shape):
    """Centre an image in a target shape by cropping or zero-padding it."""
    target_y, target_x = shape
    image = np.asarray(image, dtype=float)
    if image.shape[0] >= target_y and image.shape[1] >= target_x:
        y0 = (image.shape[0] - target_y) // 2
        x0 = (image.shape[1] - target_x) // 2
        return np.array(image[y0:y0 + target_y, x0:x0 + target_x], copy=True)
    output = np.zeros(shape, dtype=float)
    src_y0, src_x0 = max((image.shape[0] - target_y) // 2, 0), max((image.shape[1] - target_x) // 2, 0)
    dst_y0, dst_x0 = max((target_y - image.shape[0]) // 2, 0), max((target_x - image.shape[1]) // 2, 0)
    height, width = min(image.shape[0], target_y), min(image.shape[1], target_x)
    output[dst_y0:dst_y0 + height, dst_x0:dst_x0 + width] = image[src_y0:src_y0 + height, src_x0:src_x0 + width]
    return output


def plate_scale_arcsec_per_pixel(header):
    """Calculate the effective WCS plate scale in arcsec/pixel from the CD matrix."""
    try:
        cd11, cd12 = float(header["CD1_1"]), float(header["CD1_2"])
        cd21, cd22 = float(header["CD2_1"]), float(header["CD2_2"])
    except KeyError as error:
        raise KeyError(f"The mosaic is missing the required WCS keyword {error.args[0]}.") from error
    x_scale_deg, y_scale_deg = np.hypot(cd11, cd21), np.hypot(cd12, cd22)
    if x_scale_deg <= 0 or y_scale_deg <= 0:
        raise ValueError("The mosaic CD matrix does not define a valid plate scale.")
    return np.sqrt(x_scale_deg * y_scale_deg) * 3600.0


def model_file(filter_name, oversampling):
    """Find the STPSF/WebbPSF file for a filter and oversampling factor."""
    matches = sorted(SIM_PSF_DIR.glob(f"*_{filter_name.lower()}_os{oversampling}.fits"))
    if len(matches) > 1:
        raise RuntimeError(f"More than one simulated PSF matches {filter_name}: {matches}")
    return matches[0] if matches else None


def read_and_match_model(path, target_shape, target_pixel_scale):
    """Resample, centre, and normalize a simulated PSF on the mosaic grid."""
    with fits.open(path) as hdul:
        model = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    try:
        model_pixel_scale = float(header["PIXELSCL"])
    except KeyError as error:
        raise KeyError(f"{path.name} is missing the PIXELSCL header keyword.") from error
    if model.ndim != 2 or model_pixel_scale <= 0:
        raise ValueError(f"Invalid simulated PSF: {path}")
    scale_factor = model_pixel_scale / target_pixel_scale
    if not np.isclose(scale_factor, 1.0):
        model = zoom(model, scale_factor, order=3, mode="nearest", prefilter=True)
    return normalize_psf(center_match_shape(model, target_shape)), model_pixel_scale


def hybrid_psf(empirical, model, r1, r2):
    """Blend empirical and simulated PSFs with a linear radial transition."""
    if empirical.shape != model.shape:
        raise ValueError("The empirical and simulated PSFs must have the same shape.")
    if not 0 <= r1 < r2:
        raise ValueError("Hybrid radii must satisfy 0 <= r1 < r2.")
    empirical, model = normalize_psf(empirical), normalize_psf(model)
    y, x = np.indices(empirical.shape)
    radius = np.hypot(x - (empirical.shape[1] - 1) / 2, y - (empirical.shape[0] - 1) / 2)
    empirical_weight = np.ones_like(radius)
    transition = (radius > r1) & (radius < r2)
    empirical_weight[transition] = (r2 - radius[transition]) / (r2 - r1)
    empirical_weight[radius >= r2] = 0.0
    return normalize_psf(empirical_weight * empirical + (1.0 - empirical_weight) * model)


def make_header(filter_name, region_file, stars, method, psf_size, oversampling,
                image_header, model_path, model_pixel_scale, target_pixel_scale, r1, r2):
    """Create a FITS header that records hybrid-PSF provenance and parameters."""
    header = fits.Header()
    header["FILTER"] = (filter_name.upper(), "Image filter")
    header["NSTARS"] = (len(stars), "Number of stars used")
    header["PSFSIZE"] = (psf_size, "Saved PSF size (pixels)")
    header["OVERSAMP"] = (oversampling, "PSF oversampling factor")
    header["NATVSIZE"] = (SAVED_PSF_SIZE, "PSF size before oversampling")
    header["CUTSIZE"] = (CUTOUT_SIZE, "Initial cutout size (pixels)")
    header["APRAD"] = (APERTURE_RADIUS, "Star normalization radius (pixels)")
    header["METHOD"] = (method, "Stacking method")
    header["REGFILE"] = (region_file.name, "DS9 star-region file")
    header["PSFTYPE"] = ("hybrid", "Empirical core and simulated model wings")
    header["MODFILE"] = (model_path.name, "Simulated PSF file")
    header["MODPSCL"] = (model_pixel_scale, "Model pixel scale (arcsec/pixel)")
    header["TGTPSCL"] = (target_pixel_scale, "Hybrid PSF pixel scale (arcsec/pixel)")
    header["HYB1AS"] = (HYBRID_R1_ARCSEC, "Transition start radius (arcsec)")
    header["HYB2AS"] = (HYBRID_R2_ARCSEC, "Transition end radius (arcsec)")
    header["HYBR1"] = (r1, "Transition start radius (pixels)")
    header["HYBR2"] = (r2, "Transition end radius (pixels)")
    header["CD1_1"] = (float(image_header["CD1_1"]), "Original mosaic WCS CD matrix")
    header["CD2_2"] = (float(image_header["CD2_2"]), "Original mosaic WCS CD matrix")
    header["BUNIT"] = ("normalized", "PSF total flux equals one")
    for index, star in enumerate(stars.itertuples(index=False), start=1):
        header[f"S{index:03d}X"] = (float(star.x), f"Star {index} X coordinate (pixels)")
        header[f"S{index:03d}Y"] = (float(star.y), f"Star {index} Y coordinate (pixels)")
    return header


def finish_figure(figure, filter_name, suffix):
    """Save or display a diagnostic figure according to the global settings."""
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        figure.savefig(FIGURE_DIR / f"{filter_name}_{suffix}.png", dpi=180, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(figure)


def add_arcsec_axis(axis, pixel_scale):
    """Add a top horizontal axis that shows radius in arcsec."""
    arcsec_axis = axis.secondary_xaxis(
        "top", functions=(lambda pixels: pixels * pixel_scale, lambda arcsec: arcsec / pixel_scale)
    )
    arcsec_axis.set_xlabel("Radius (arcsec)")


def save_diagnostic_figures(filter_name, cutouts_norm, cutouts_centered,
                            empirical, model, hybrid, r1, r2, pixel_scale):
    """Save cutout, centering, PSF, radial-profile, and encircled-energy diagnostics."""
    names, ncols = list(cutouts_norm), 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, name in zip(axes.ravel(), names):
        image = cutouts_norm[name]
        axis.imshow(image, origin="lower", cmap="gray", norm=simple_norm(image, "log", percent=99.8))
        axis.set_title(str(name), fontsize=9)
    fig.suptitle(f"{filter_name}: normalized stars")
    fig.tight_layout()
    finish_figure(fig, filter_name, "normalized_cutouts")

    centers = measure_centers(cutouts_centered)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, name in zip(axes.ravel(), names):
        image = cutouts_centered[name]
        ny, nx = image.shape
        axis.imshow(image, origin="lower", cmap="gray", norm=simple_norm(image, "log", percent=99.5))
        axis.plot((nx - 1) / 2, (ny - 1) / 2, "+", color="red", ms=10)
        axis.plot(centers[name]["x"], centers[name]["y"], "o", color="cyan", ms=4)
        axis.set_title(str(name), fontsize=9)
    fig.suptitle(f"{filter_name}: centering")
    fig.tight_layout()
    finish_figure(fig, filter_name, "centers")

    psfs = ((empirical, "Empirical"), (model, "STPSF/WebbPSF"), (hybrid, "Hybrid"))
    fig, axes = plt.subplots(1, len(psfs), figsize=(15, 5))
    for axis, (image, title) in zip(axes, psfs):
        axis.imshow(image, origin="lower", cmap="gray", norm=simple_norm(image, "log", percent=99.5))
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"PSF {filter_name}: transition {r1:.2f}-{r2:.2f} pixels "
        f"({HYBRID_R1_ARCSEC:.2f}-{HYBRID_R2_ARCSEC:.2f} arcsec)"
    )
    fig.tight_layout()
    finish_figure(fig, filter_name, "hybrid_psf")

    fig, axis = plt.subplots(figsize=(6, 4))
    for image, label in psfs:
        radius, profile = radial_profile(image)
        valid = profile > 0
        axis.plot(radius[valid], profile[valid], label=label)
    axis.axvline(r1, color="black", ls="--", lw=1, label="Transition range")
    axis.axvline(r2, color="black", ls="--", lw=1)
    axis.set_yscale("log")
    axis.set(xlabel="Radius (pixels)", ylabel="Mean flux per pixel", title=f"Radial profile: {filter_name}")
    axis.legend()
    add_arcsec_axis(axis, pixel_scale)
    fig.tight_layout()
    finish_figure(fig, filter_name, "hybrid_radial_profile")

    fig, axis = plt.subplots(figsize=(6, 4))
    for image, label in psfs:
        radius, energy = encircled_energy(image)
        axis.plot(radius, energy, label=label)
    axis.set(xlabel="Radius (pixels)", ylabel="Encircled energy", ylim=(0, 1.05),
             title=f"Encircled energy: {filter_name}")
    axis.grid()
    axis.legend()
    add_arcsec_axis(axis, pixel_scale)
    fig.tight_layout()
    finish_figure(fig, filter_name, "hybrid_encircled_energy")


def build_hybrid_psf(filter_name):
    """Build and write native and two-times-oversampled hybrid PSFs for one filter."""
    image_file = DATA_DIR / f"abells1063-grizli-v7.4-{filter_name}_sci.fits"
    region_file = DATA_DIR / REGION_FILE_OVERRIDES.get(filter_name, f"stars_{filter_name}_new.reg")
    if not region_file.exists():
        region_file = DATA_DIR / f"stars_{filter_name}.reg"
    native_model_file = model_file(filter_name, oversampling=1)
    oversampled_model_file = model_file(filter_name, oversampling=OVERSAMPLING)
    if native_model_file is None or oversampled_model_file is None:
        print(f"{filter_name}: skipped (missing os1/os{OVERSAMPLING} simulated PSF).")
        return None
    if not image_file.exists() or not region_file.exists():
        print(f"{filter_name}: skipped (missing science FITS or DS9 region file).")
        return None
    stars = read_ds9_boxes(region_file)
    if stars.empty:
        print(f"{filter_name}: skipped (no stars in the DS9 region file).")
        return None

    with fits.open(image_file) as hdul:
        cutouts = extract_cutouts(hdul[0].data, stars, size=CUTOUT_SIZE)
        image_header = hdul[0].header.copy()
    native_pixel_scale = plate_scale_arcsec_per_pixel(image_header)
    os2_pixel_scale = native_pixel_scale / OVERSAMPLING
    native_r1, native_r2 = HYBRID_R1_ARCSEC / native_pixel_scale, HYBRID_R2_ARCSEC / native_pixel_scale
    os2_r1, os2_r2 = HYBRID_R1_ARCSEC / os2_pixel_scale, HYBRID_R2_ARCSEC / os2_pixel_scale

    cutouts_sub, background_stds = {}, {}
    for name, cutout in cutouts.items():
        cutouts_sub[name], _, background_stds[name] = subtract_background(cutout.data, border=BACKGROUND_BORDER)
    cutouts_centered = recenter_cutouts(cutouts_sub, measure_centers(cutouts_sub))
    cutouts_norm, fluxes = normalize_by_aperture(cutouts_centered, radius=APERTURE_RADIUS)
    if filter_name.lower() in WEIGHTED_SIGMA_CLIPPED_FILTERS:
        weights = {name: fluxes[name] / background_stds[name] for name in fluxes}
        empirical_full = weighted_sigma_clipped_psf(cutouts_norm, weights, sigma=SIGMA_CLIP)
        method = "weighted_sigma_clipped"
    else:
        empirical_full = np.nanmedian(np.array(list(cutouts_norm.values())), axis=0)
        method = "median"

    empirical_native = normalize_psf(center_crop(empirical_full, SAVED_PSF_SIZE))
    empirical_os2 = normalize_psf(zoom(empirical_native, OVERSAMPLING, order=3, mode="nearest", prefilter=True))
    model_native, model_native_scale = read_and_match_model(native_model_file, empirical_native.shape, native_pixel_scale)
    model_os2, model_os2_scale = read_and_match_model(oversampled_model_file, empirical_os2.shape, os2_pixel_scale)
    hybrid_native = hybrid_psf(empirical_native, model_native, native_r1, native_r2)
    hybrid_os2 = hybrid_psf(empirical_os2, model_os2, os2_r1, os2_r2)

    output = DATA_DIR / f"hybrid_psf_filter_{filter_name}.fits"
    header = make_header(filter_name, region_file, stars, method, SAVED_PSF_SIZE, 1, image_header,
                         native_model_file, model_native_scale, native_pixel_scale, native_r1, native_r2)
    fits.writeto(output, hybrid_native, header=header, overwrite=True)
    output_os2 = DATA_DIR / f"hybrid_psf_filter_{filter_name}_os{OVERSAMPLING}.fits"
    header_os2 = make_header(filter_name, region_file, stars, method, SAVED_PSF_SIZE * OVERSAMPLING,
                             OVERSAMPLING, image_header, oversampled_model_file, model_os2_scale,
                             os2_pixel_scale, os2_r1, os2_r2)
    fits.writeto(output_os2, hybrid_os2, header=header_os2, overwrite=True)
    print(f"{filter_name}: {len(stars)} stars; method={method}; wrote {output.name} and {output_os2.name}")

    if SAVE_FIGURES or SHOW_FIGURES:
        save_diagnostic_figures(filter_name, cutouts_norm, cutouts_centered, empirical_native,
                                model_native, hybrid_native, native_r1, native_r2, native_pixel_scale)
    return hybrid_native


if __name__ == "__main__":
    for current_filter in FILTERS:
        build_hybrid_psf(current_filter)
