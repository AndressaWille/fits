import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import re


def get_filter(fname):
    """
    Extracts the filter name from a FITS filename using a regular expression.

    Parameters
    ----------
    fname : str
        FITS filename.

    Returns
    -------
    filt : str
        Filter identifier (e.g., 'F150W', 'F410M').

    Raises
    ------
    ValueError
        If no filter name is found in the filename.
    """
    m = re.search(r'F\d{3}[WNM]', fname.upper())
    if m:
        return m.group(0)
    raise ValueError(f"Filter not found in filename: {fname}")

def find_ext(hdul):
    """
    Finds the first FITS extension containing valid image data.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Opened FITS file.

    Returns
    -------
    ext : int or None
        Index of the HDU containing image data. Returns None if not found.
    """
    for i, hdu in enumerate(hdul):
        if hdu.data is not None and isinstance(hdu.data, np.ndarray):
            return i
    return None

def get_pixel_scale(fits_path):
    """
    Returns PSF pixel scale in arcsec/pixel and data shape.
    """
    with fits.open(fits_path) as hdul_ref:
        ext_ref = find_ext(hdul_ref)
        if ext_ref is None:
            raise ValueError(f"No valid image data in reference file '{ref_file}'.")
        ref_header = hdul_ref[ext_ref].header
        data = hdul_ref[0].data
        w = WCS(ref_header)
        pixel_scale = np.abs(w.wcs.cd[1,1]) * 3600  #deg/pixel --> arcsec/pixel
        return pixel_scale, pixel_scale, data.shape

def build_kernel(psf_source, psf_target, shape, eps=1e-3):
    """
    Build a convolution kernel that transforms psf_source into psf_target
    using Fourier methods.

    Parameters
    ----------
    psf_source : 2D array
        PSF of the original image.
    psf_target : 2D array
        PSF of the target resolution.
    shape : tuple
        Shape (Ny, Nx) of the final kernel/image.
    eps : float
        Regularization parameter to avoid division by zero.

    Returns
    -------
    kernel : 2D array
        Convolution kernel.
    """
    Ny, Nx = shape

    psf_s = np.zeros((Ny, Nx))
    psf_t = np.zeros((Ny, Nx))

    psf_s[:psf_source.shape[0], :psf_source.shape[1]] = psf_source
    psf_t[:psf_target.shape[0], :psf_target.shape[1]] = psf_target

    F_s = fft2(psf_s)
    F_t = fft2(psf_t)

    F_kernel = F_t * np.conj(F_s) / (np.abs(F_s)**2 + eps)

    kernel = np.real(ifft2(F_kernel))
    kernel = fftshift(kernel)
    kernel /= np.sum(kernel)

    return kernel

def save_kernel(kernel, output_path, header=None):
    """
    Save a convolution kernel to a FITS file.

    Parameters
    ----------
    kernel : 2D array
        Convolution kernel.
    output_path : str
        Output FITS path.
    header : fits.Header or None
        Optional header to attach to the kernel.
    """
    hdu = fits.PrimaryHDU(kernel, header=header)
    hdu.writeto(output_path, overwrite=True)
    print(f"Kernel saved at {output_path}")


def apply_kernel(image, kernel):
    """
    Convolve an image with a given kernel.

    Parameters
    ----------
    image : 2D array
    kernel : 2D array

    Returns
    -------
    image_conv : 2D array
    """
    return convolve_fft(image, kernel, allow_huge=True, normalize_kernel=False)