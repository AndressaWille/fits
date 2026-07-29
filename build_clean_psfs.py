#!/usr/bin/env python3
"""Build empirical PSFs, optionally applying post-processing to their wings.

The workflow extracts stellar cutouts, subtracts their backgrounds, recenters
them, normalizes them in a circular aperture, and stacks them. With
``APPLY_CLEANING`` enabled, residual background is removed and only pixels
outside ``CLEAN_RADIUS_ARCSEC`` are Gaussian-smoothed.
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
from scipy.ndimage import gaussian_filter, shift, zoom


DATA_DIR = Path(__file__).resolve().parent
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

# Set to False to save the empirical stacked PSF without post-processing.
APPLY_CLEANING = True
CLEAN_RADIUS_ARCSEC = 1.0
CLEAN_SIGMA = 1
CLEAN_SUBTRACT_BACKGROUND = True
CLEAN_ENFORCE_SYMMETRY = False

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
    """Extract square cutouts centred on selected stars."""
    return {
        star.name: Cutout2D(image, (star.x, star.y), size=size, mode="partial", fill_value=np.nan)
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
    """Measure quadratic centroids for a dictionary of images."""
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
    """Normalize each cutout by its flux in a central circular aperture."""
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
    """Return mean flux in successive one-pixel radial annuli."""
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


def center_crop(image, size):
    """Extract a square crop centred on an image."""
    ny, nx = image.shape
    if size > ny or size > nx:
        raise ValueError(f"Cannot extract a {size}x{size} crop from a {nx}x{ny} PSF.")
    y0, x0 = (ny - size) // 2, (nx - size) // 2
    return np.array(image[y0:y0 + size, x0:x0 + size], copy=True)


def normalize_psf(psf):
    """Replace invalid values, clip negatives, and normalize total flux to one."""
    psf = np.array(psf, dtype=float, copy=True)
    psf[~np.isfinite(psf)] = 0.0
    psf[psf < 0] = 0.0
    total = psf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"The PSF cannot be normalized; total={total}.")
    return psf / total


def oversample_psf(psf, factor, order=3):
    """Interpolate a PSF by an integer factor and renormalize it."""
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("The oversampling factor must be a positive integer.")
    return normalize_psf(zoom(psf, zoom=factor, order=order, mode="nearest", prefilter=True))


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


def clean_psf(psf, core_radius, sigma, subtract_background, enforce_symmetry):
    """Remove residual background and smooth only the PSF wings."""
    cleaned = np.array(psf, dtype=float, copy=True)
    cleaned[~np.isfinite(cleaned)] = 0.0
    if subtract_background:
        edge = np.concatenate([cleaned[0, :], cleaned[-1, :], cleaned[:, 0], cleaned[:, -1]])
        cleaned -= np.nanmedian(edge)
    cleaned[cleaned < 0] = 0.0
    y, x = np.indices(cleaned.shape)
    radius = np.hypot(x - (cleaned.shape[1] - 1) / 2, y - (cleaned.shape[0] - 1) / 2)
    smoothed = gaussian_filter(cleaned, sigma=sigma)
    cleaned[radius > core_radius] = smoothed[radius > core_radius]
    if enforce_symmetry:
        cleaned = sum(np.rot90(cleaned, turns) for turns in range(4)) / 4
    return cleaned


def make_header(filter_name, region_file, stars, method, psf_size, oversampling,
                image_header, clean_radius_pixels):
    """Create a FITS header with PSF provenance and processing parameters."""
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
    header["CLEANED"] = (APPLY_CLEANING, "Post-processing cleaning applied")
    header["CLNRAD"] = (CLEAN_RADIUS_ARCSEC, "Cleaning start radius (arcsec)")
    header["CLNCORE"] = (clean_radius_pixels, "Unsmooth core radius (pixels)")
    header["CLNSIG"] = (CLEAN_SIGMA, "Gaussian wing-smoothing sigma")
    header["CLNBG"] = (CLEAN_SUBTRACT_BACKGROUND, "Residual background subtraction")
    header["CLNSYM"] = (CLEAN_ENFORCE_SYMMETRY, "Rotational symmetry applied")
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
    """Add a top horizontal axis that expresses radius in arcsec."""
    arcsec_axis = axis.secondary_xaxis(
        "top", functions=(lambda pixels: pixels * pixel_scale, lambda arcsec: arcsec / pixel_scale)
    )
    arcsec_axis.set_xlabel("Radius (arcsec)")


def save_diagnostic_figures(filter_name, cutouts_norm, cutouts_centered,
                            psf_raw, psf_final, pixel_scale, clean_radius_pixels):
    """Save cutout, centering, PSF, radial-profile, and energy diagnostics."""
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

    final_title = "cleaned and normalized" if APPLY_CLEANING else "normalized without cleaning"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, image, title in zip(axes, (psf_raw, psf_final), ("stacked (101 pixels)", final_title)):
        axis.imshow(image, origin="lower", cmap="gray", norm=simple_norm(image, "log", percent=99.5))
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"PSF {filter_name}")
    fig.tight_layout()
    finish_figure(fig, filter_name, "psf")

    fig, axis = plt.subplots(figsize=(6, 4))
    for image, label in ((psf_raw, "stacked"), (psf_final, "final")):
        radius, profile = radial_profile(image)
        valid = profile > 0
        axis.plot(radius[valid], profile[valid], label=label)
    if APPLY_CLEANING:
        axis.axvline(clean_radius_pixels, color="black", ls="--", lw=1, label="Cleaning start")
    axis.set_yscale("log")
    axis.set(xlabel="Radius (pixels)", ylabel="Mean flux per pixel", title=f"Radial profile: {filter_name}")
    axis.legend()
    add_arcsec_axis(axis, pixel_scale)
    fig.tight_layout()
    finish_figure(fig, filter_name, "radial_profile")

    fig, axis = plt.subplots(figsize=(6, 4))
    radius, energy = encircled_energy(psf_final)
    axis.plot(radius, energy)
    axis.set(xlabel="Radius (pixels)", ylabel="Encircled energy", ylim=(0, 1.05),
             title=f"Encircled energy: {filter_name}")
    axis.grid()
    add_arcsec_axis(axis, pixel_scale)
    fig.tight_layout()
    finish_figure(fig, filter_name, "encircled_energy")


def build_clean_psf(filter_name):
    """Build and write native and two-times-oversampled empirical PSFs for one filter."""
    image_file = DATA_DIR / f"abells1063-grizli-v7.4-{filter_name}_sci.fits"
    region_file = DATA_DIR / REGION_FILE_OVERRIDES.get(filter_name, f"stars_{filter_name}_new.reg")
    if not region_file.exists():
        region_file = DATA_DIR / f"stars_{filter_name}.reg"
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
    pixel_scale = plate_scale_arcsec_per_pixel(image_header)
    clean_radius_pixels = CLEAN_RADIUS_ARCSEC / pixel_scale

    cutouts_sub, background_stds = {}, {}
    for name, cutout in cutouts.items():
        cutouts_sub[name], _, background_stds[name] = subtract_background(cutout.data, border=BACKGROUND_BORDER)
    cutouts_centered = recenter_cutouts(cutouts_sub, measure_centers(cutouts_sub))
    cutouts_norm, fluxes = normalize_by_aperture(cutouts_centered, radius=APERTURE_RADIUS)
    if filter_name.lower() in WEIGHTED_SIGMA_CLIPPED_FILTERS:
        weights = {name: fluxes[name] / background_stds[name] for name in fluxes}
        psf_full = weighted_sigma_clipped_psf(cutouts_norm, weights, sigma=SIGMA_CLIP)
        method = "weighted_sigma_clipped"
    else:
        psf_full = np.nanmedian(np.array(list(cutouts_norm.values())), axis=0)
        method = "median"

    psf_raw = center_crop(psf_full, SAVED_PSF_SIZE)
    psf_final = clean_psf(psf_raw, clean_radius_pixels, CLEAN_SIGMA,
                          CLEAN_SUBTRACT_BACKGROUND, CLEAN_ENFORCE_SYMMETRY) if APPLY_CLEANING else psf_raw
    psf_final = normalize_psf(psf_final)
    psf_oversampled = oversample_psf(psf_final, factor=OVERSAMPLING)
    output_tag = "clean" if APPLY_CLEANING else "raw"
    output = DATA_DIR / f"empirical_psf_{output_tag}_filter_{filter_name}.fits"
    header = make_header(filter_name, region_file, stars, method, SAVED_PSF_SIZE, 1,
                         image_header, clean_radius_pixels)
    fits.writeto(output, psf_final, header=header, overwrite=True)
    output_os2 = DATA_DIR / f"empirical_psf_{output_tag}_filter_{filter_name}_os{OVERSAMPLING}.fits"
    header_os2 = make_header(filter_name, region_file, stars, method, SAVED_PSF_SIZE * OVERSAMPLING,
                             OVERSAMPLING, image_header, clean_radius_pixels * OVERSAMPLING)
    fits.writeto(output_os2, psf_oversampled, header=header_os2, overwrite=True)
    state = "enabled" if APPLY_CLEANING else "disabled"
    print(f"{filter_name}: {len(stars)} stars; method={method}; cleaning={state}; "
          f"plate scale={pixel_scale:.6f} arcsec/pixel; wrote {output.name} and {output_os2.name}")

    if SAVE_FIGURES or SHOW_FIGURES:
        save_diagnostic_figures(filter_name, cutouts_norm, cutouts_centered, psf_raw, psf_final,
                                pixel_scale, clean_radius_pixels)
    return psf_final


if __name__ == "__main__":
    for current_filter in FILTERS:
        build_clean_psf(current_filter)
