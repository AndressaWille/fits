## functions to cut and create datacubes with photometric data from HST and JWST

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regions import Regions
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from astropy.visualization import simple_norm



# Filter sets

NIRCAM = {"F070W","F090W","F115W","F140M","F150W","F162M","F164N","F187N","F182M","F200W","F210M",
          "F250M","F277W","F300M","F335M","F356W","F360M","F410M","F430M","F444W","F460M","F480M"}
ACS = {"F435W","F475W","F606W","F625W","F814W"}
WFC3_IR = {"F105W","F110W","F125W","F140W","F160W"}
WFC3_UV = {"F225W","F275W","F336W"}


def pick_folder(filt):
    """
    Returns the instrument folder name based on the filter.
    """
    if filt in NIRCAM:
        return "NIRCam"
    if filt in ACS:
        return "ACS"
    if filt in WFC3_IR:
        return "WFC3-IR"
    if filt in WFC3_UV:
        return "WFC3-UV"
    return "OTHERS"


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


def infer_filter(fname):
    """
    Infers the filter name from a FITS filename.

    Parameters
    ----------
    fname : str
        FITS filename.

    Returns
    -------
    filt : str
        Inferred filter name.
    """
    base = os.path.basename(fname).upper().replace('.', '_')
    for part in base.split('_'):
        if part.startswith('F') and any(c.isdigit() for c in part):
            return part
    return os.path.splitext(os.path.basename(fname))[0]


def filter_id(filt):
    """
    Extracts the numeric part of a filter name for sorting.

    Parameters
    ----------
    filt : str
        Filter name.

    Returns
    -------
    num : int
        Numeric wavelength identifier. If not found, returns a large value
        so the filter is sorted last.
    """
    m = re.search(r'F(\d+)', filt)
    return int(m.group(1)) if m else 9999



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


def crop_fits(fits_path, region):
    """
    Crops a FITS image using a DS9 region.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    region : regions.Region
        Region object read from a DS9 region file.

    Returns
    -------
    data_cut : numpy.ndarray
        Cropped image data.
    header_cut : astropy.io.fits.Header
        Updated FITS header with adjusted CRPIX keywords.

    Raises
    ------
    ValueError
        If no image data extension is found in the FITS file.
    """
    # Open fits
    with fits.open(fits_path) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No image data found in {fits_path}")

        data = hdul[ext].data
        header = hdul[ext].header
        wcs = WCS(header)
    
    # Define region limits
    pix_region = region.to_pixel(wcs)
    bbox = pix_region.bounding_box

    x_min, x_max = int(bbox.ixmin), int(bbox.ixmax)
    y_min, y_max = int(bbox.iymin), int(bbox.iymax)

    # Crop the image
    data_cut = data[y_min:y_max, x_min:x_max]

    # Update reference pixel in the header
    header_cut = header.copy()
    header_cut['CRPIX1'] -= x_min
    header_cut['CRPIX2'] -= y_min

    return data_cut, header_cut


def save_fits(data, header, path):
    """
    Saves a FITS file to disk.

    Parameters
    ----------
    data : numpy.ndarray
        Image data.
    header : astropy.io.fits.Header
        FITS header.
    path : str
        Output FITS path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


def sort_fits(folder):
    """
    Reads all FITS files in a folder and sorts them by filter wavelength.

    Parameters
    ----------
    folder : str
        Path to the folder containing FITS files.

    Returns
    -------
    fits_sorted : list of tuples
        List in the form [(fits_path, filter_name), ...], sorted by filter id.
    """
    fits_files = glob.glob(os.path.join(folder, '*.fits'))
    fits_list = [(f, infer_filter(f)) for f in fits_files]
    fits_sorted = sorted(fits_list, key=lambda x: filter_id(x[1]))
    return fits_sorted


def align_fits(fits_list, ref_file):
    """
    Aligns FITS images to a common WCS using a reference FITS file.

    Parameters
    ----------
    fits_list : list of tuples
        List in the form [(fits_path, filter_name), ...].
    ref_file : str
        Path to the FITS file used as WCS reference.

    Returns
    -------
    aligned_list : list of tuples
        List in the form [(aligned_fits_path, filter_name), ...].

    """
    # Open reference FITS
    with fits.open(ref_file) as hdul_ref:
        ext_ref = find_ext(hdul_ref)
        if ext_ref is None:
            raise ValueError(f"No valid image data in reference file '{ref_file}'.")
        ref_header = hdul_ref[ext_ref].header

    aligned_list = []

    # Loop over all FITS to be aligned
    for fpath, filt in fits_list:

        with fits.open(fpath) as hdul:
            ext = find_ext(hdul)
            if ext is None:
                print(f"No data extension found in {fpath}. Skipping.")
                continue

            data = hdul[ext].data
            wcs_in = WCS(hdul[ext].header)

            unit = hdul[ext].header.get('BUNIT', 'unknown')
            print(f"Filter {filt}: unit = {unit}")

            # Reproject to reference WCS
            data_aligned, _ = reproject_interp((data, wcs_in), ref_header)

            # Build aligned header
            header_aligned = ref_header.copy()
            header_aligned.update(WCS(ref_header).to_header())

            # Output name
            out_name = os.path.splitext(fpath)[0] + "_aligned.fits"

            # Save
            fits.PrimaryHDU(data_aligned, header=header_aligned).writeto(
                out_name, overwrite=True
            )

            aligned_list.append((out_name, filt))
            print(f"Saved: {out_name}")

    return aligned_list  
    

def MJy_sr_to_jy(aligned_list):
    """
    Convert FITS images from MJy/sr to Jy/pixel using the PIXAR_SR keyword: Jy/pixel = (MJy/sr) * 1e6 * PIXAR_SR

    Parameters
    ----------
    aligned_list : list of tuples
        List in the form [(fits_path, filter_name), ...].

    Returns
    -------
    new_list : list of tuples
        List in the form [(new_fits_path, filter_name), ...] for the converted files.

    """
    new_list = []

    for fpath, filt in aligned_list:

        with fits.open(fpath) as hdul:
            ext = find_ext(hdul)
            if ext is None:
                print(f"No data extension found in '{fpath}'. Skipping.")
                continue

            data = hdul[ext].data.astype(np.float64)
            header = hdul[ext].header.copy()

            # Pixel area in steradians
            pixar_sr = header.get('PIXAR_SR')
            if pixar_sr is None:
                print(f"PIXAR_SR not found in '{fpath}'. Skipping.")
                continue

            # Conversion factor: MJy -> Jy and multiply by pixel area
            factor = 1e6 * pixar_sr
            data_jy = data * factor

            # Update header
            header['BUNIT'] = 'Jy'
            header.add_history(
                f"Converted from MJy/sr to Jy/pixel using PIXAR_SR = {pixar_sr}"
            )

            # Output file name
            out_name = os.path.splitext(fpath)[0] + "_Jy.fits"

            # Save
            fits.PrimaryHDU(data_jy, header=header).writeto(out_name, overwrite=True)
            print(f"Saved: {out_name}")

            new_list.append((out_name, filt))

    return new_list



def build_datacube(aligned_fits_files, reference_file, output_path):
    """
    Builds a 3D datacube from aligned 2D FITS images.

    Parameters
    ----------
    aligned_fits_files : list of tuples
        [(filename, filter_name), ...]
    reference_file : str
        FITS file to define WCS and shape.
    output_path : str
        Path to save the 3D datacube.
    """

    with fits.open(reference_file) as hdul_ref:
        ext_ref = find_ext(hdul_ref)
        if ext_ref is None:
            raise ValueError(f"No 2D data in reference file {reference_file}.")
        ref_header = hdul_ref[ext_ref].header
        ny, nx = hdul_ref[ext_ref].data.shape
        wcs_2d = WCS(ref_header, naxis=2)

    cube_images = []
    filter_names = []
    units = []

    for file, filt in aligned_fits_files:
        with fits.open(file) as hdul:
            ext = find_ext(hdul)
            if ext is None:
                print(f"No 2D data in {file}. Skipping.")
                continue
            data = hdul[ext].data
            cube_images.append(data)
            filter_names.append(filt)
            units.append(hdul[ext].header.get('BUNIT', 'unknown'))

    cube = np.array(cube_images)

    wcs_3d = WCS(naxis=3)
    wcs_3d.wcs.crpix[0] = wcs_2d.wcs.crpix[0]
    wcs_3d.wcs.crpix[1] = wcs_2d.wcs.crpix[1]
    wcs_3d.wcs.crval[0] = wcs_2d.wcs.crval[0]
    wcs_3d.wcs.crval[1] = wcs_2d.wcs.crval[1]
    wcs_3d.wcs.cdelt[0] = wcs_2d.wcs.cdelt[0]
    wcs_3d.wcs.cdelt[1] = wcs_2d.wcs.cdelt[1]
    wcs_3d.wcs.ctype[0] = wcs_2d.wcs.ctype[0]
    wcs_3d.wcs.ctype[1] = wcs_2d.wcs.ctype[1]
    wcs_3d.wcs.cunit[0] = wcs_2d.wcs.cunit[0]
    wcs_3d.wcs.cunit[1] = wcs_2d.wcs.cunit[1]

    if wcs_2d.wcs.has_cd():
        cd3 = np.zeros((3,3))
        cd3[0:2,0:2] = wcs_2d.wcs.cd.copy()
        cd3[2,2] = 1.0
        wcs_3d.wcs.cd = cd3

    wcs_3d.wcs.crpix[2] = 1
    wcs_3d.wcs.crval[2] = 0
    wcs_3d.wcs.cdelt[2] = 1
    wcs_3d.wcs.ctype[2] = 'FILTER'
    wcs_3d.wcs.cunit[2] = ''

    header = wcs_3d.to_header()
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = len(filter_names)

    if all(u == units[0] for u in units):
        header['BUNIT'] = units[0]
    else:
        header['BUNIT'] = 'unknown'

    for i, filt in enumerate(filter_names):
        header[f'FILTER{i+1}'] = filt

    fits.PrimaryHDU(cube, header=header).writeto(output_path, overwrite=True)
    print(f"Datacube saved at {output_path}")

          
        
def visualize_fits(fits_path, save_path=None, stretch='log',
                   min_percent=25., max_percent=99.98,
                   cmap='viridis', xlim=None, ylim=None):
    """
    Visualizes a FITS file with both pixel and RA/Dec axes.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    save_path : str or None
        Path to save the output image. If None, the figure is shown but not saved.
    stretch : str
        Stretch type for simple_norm (e.g., 'linear', 'log', 'sqrt').
    min_percent, max_percent : float
        Percentile limits for normalization.
    cmap : str
        Colormap name.
    xlim, ylim : tuple or None
        Pixel axis limits.
    """
    # Open FITS file
    with fits.open(fits_path, memmap=False) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            print(f"No data extension found in {fits_path}")
            return

        data = hdul[ext].data
        header = hdul[ext].header
        wcs = WCS(header)

    # Mask invalid or non-positive values for normalization
    mask = np.logical_or(np.isnan(data), data <= 0.)
    norm = simple_norm(
        data[~mask],
        stretch=stretch,
        min_percent=min_percent,
        max_percent=max_percent
    )

    # Create figure
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': wcs})

    im = ax.imshow(
        data,
        cmap=cmap,
        origin='lower',
        norm=norm,
        interpolation='nearest'
    )

    # World coordinates
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    # Pixel coordinates
    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('x (pixel)')
    secay = ax.secondary_yaxis('right')
    secay.set_ylabel('y (pixel)')

    # Set limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.grid()

    # Use filename to build a simple title
    base = os.path.basename(fits_path)
    region = base.split('_')[0]
    ax.set_title(region)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved image at {save_path}")
        plt.close(fig)
    else:
        plt.show()


def cut_region(cube_fits_file, x_start, x_end, y_start, y_end, output_path):
    """
    Cuts a spatial region from a datacube.

    Parameters
    ----------
    cube_fits_file : str
        Path to the input 3D fits datacube.
    x_start, x_end : int
        Pixel indices for the x axis.
    y_start, y_end : int
        Pixel indices for the y axis.
    output_filename : str
        Path to the output fits file.

    Returns
    -------
    None
        Saves the cut datacube.
    """
    # Open datacube
    with fits.open(cube_fits_file) as hdul:
        cube_data = hdul[0].data
        cube_header = hdul[0].header

        # Cut the data array
        cut_data = cube_data[:, y_start:y_end, x_start:x_end]

        # Get and update WCS
        wcs_3d = WCS(cube_header)
        wcs_3d.wcs.crpix[0] -= x_start
        wcs_3d.wcs.crpix[1] -= y_start

        # Create new header with updated size and WCS
        new_header = wcs_3d.to_header()
        new_header['NAXIS'] = 3
        new_header['NAXIS1'] = x_end - x_start
        new_header['NAXIS2'] = y_end - y_start
        new_header['NAXIS3'] = cube_data.shape[0]

        # Filter info
        for key in cube_header:
            if key.startswith('FILTER'):
                new_header[key] = cube_header[key]

        # Crop window
        new_header['XMINPIX'] = x_start
        new_header['XMAXPIX'] = x_end
        new_header['YMINPIX'] = y_start
        new_header['YMAXPIX'] = y_end

        # Save cut datacube
        fits.PrimaryHDU(cut_data, header=new_header).writeto(output_path, overwrite=True)
        print(f"Cut datacube saved to '{output_path}'")
        
        
        
def build_valid_datacube(cube_fits_file, output_cube, threshold=0.0, frac_valid=0.01):
    """
    Remove empty filters in a datacube, saves the new datacube and returns the valid filter names.
    
    Parameters
    ----------
    cube_fits_file : str
        Path to the input 3D fits datacube.
    output_cube: str
        Path to save the filtered fits. 
    threshold : float
        Minimum flux value to consider a pixel valid.
    frac_valid : float
        Minimum fraction of pixels above the threshold to consider the filter valid.
    
    Retorna
    -------
    cube_filtered : np.ndarray
        Datacube with only valid filters. 
    filters_valid : list
        List of valid filter names.
        
    Saves the new datacube.
    """
    
    with fits.open(cube_fits_file) as hdul:
        cube = hdul[0].data
        header = hdul[0].header
        
        n_filters = header.get("NAXIS3", cube.shape[0])
        filters = [header[f"FILTER{i+1}"].strip() for i in range(n_filters)]
    
    # Keep only valid filters
    valid_indices = []
    for i, img in enumerate(cube):
        valid = np.isfinite(img)
        n_total = valid.sum()
        if n_total == 0:
            continue
        n_above = np.sum(img[valid] > threshold)
        if (n_above / n_total) >= frac_valid:
            valid_indices.append(i)

    cube_filtered = cube[valid_indices]
    filters_valid = [filters[i] for i in valid_indices]
    
    # Write new header
    new_header = header.copy()
    new_header["NAXIS3"] = len(filters_valid)

    for i, f in enumerate(filters_valid):
        new_header[f"FILTER{i+1}"] = f

    for i in range(len(filters_valid), n_filters):
        key = f"FILTER{i+1}"
        if key in new_header:
            del new_header[key]

    # Save filtered datacube 
    hdu = fits.PrimaryHDU(cube_filtered, header=new_header)
    hdu.writeto(output_cube, overwrite=True)
    print(f"Filtered datacube saved at '{output_cube}'")

    return cube_filtered, filters_valid


def remove_filter(cube_fits_file, output_cube, filter_to_remove):
    """
    Removes a specific filter from a 3D datacube.

    Parameters
    ----------
    cube_fits_file : str
        Path to the original datacube.
    output_cube : str
        Path to save the new datacube.
    filter_to_remove : str
        Name of the filter to be removed (e.g., ‘F115W’).
    """

    with fits.open(cube_fits_file) as hdul:
        cube = hdul[0].data
        header = hdul[0].header

        n_filters = header.get("NAXIS3", cube.shape[0])
        filters = [header[f"FILTER{i+1}"].strip() for i in range(n_filters)]

    # Keep filters
    keep_indices = [i for i, f in enumerate(filters) if f != filter_to_remove]

    if len(keep_indices) == len(filters):
        print(f"This filter '{filter_to_remove}' is not in the datacube.")
    else:
        print(f"Removing filter: {filter_to_remove}")

    # New cube
    cube_new = cube[keep_indices]
    filters_new = [filters[i] for i in keep_indices]

    # Write new header
    new_header = header.copy()
    new_header["NAXIS3"] = len(filters_new)

    for i, f in enumerate(filters_new):
        new_header[f"FILTER{i+1}"] = f

    for i in range(len(filters_new), n_filters):
        key = f"FILTER{i+1}"
        if key in new_header:
            del new_header[key]

    # Save
    hdu = fits.PrimaryHDU(cube_new, header=new_header)
    hdu.writeto(output_cube, overwrite=True)

    print(f"New datacube without '{filter_to_remove}' saved at {output_cube}")
    return cube_new, filters_new
