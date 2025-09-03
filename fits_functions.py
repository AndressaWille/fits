## -------- Functions to cut regions, visualize fits files, and create datacubes with many filters -------- ##

from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
from astropy.visualization import simple_norm
import numpy as np
import matplotlib.pyplot as plt
from reproject import reproject_interp
import os
import pandas as pd

# --------- # --------- #

def find_data_extension(hdul):
    """
    Returns the index of the extension containing valid image data in a fits file.
    If none is found, returns None.
    """
    for i, hdu in enumerate(hdul):
        if hdu.data is not None and isinstance(hdu.data, np.ndarray):
            return i
    return None


# --------- # --------- #

def cut_fits_with_region(fits_files, reg_file, output_dir):
    """
    Cut fits images based on regions defined in a DS9 .reg file.
    
    Parameters
    ----------
    fits_files : list
        List of paths to fits files.
    reg_file : str
        Path to the DS9 .reg file.
    output_dir : str
        Output directory to save cut fits.
        
    Returns
    -------
    None
        Saves the cut fits.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read regions from DS9
    regions_list = Regions.read(reg_file, format='ds9')

    # Cut loop for all filters
    for fits_path in fits_files:
        base = os.path.basename(fits_path)
        filter_name = os.path.splitext(base)[0]
        filter_name = filter_name.upper()  # example: 'f277w' -> 'F277W' (RENAME THE FILE WITH THE NAME OF THE FILTER AFTER YOU DOWNLOAD IT!!!)

        with fits.open(fits_path, memmap=False) as hdul:
            ext = find_data_extension(hdul)
            if ext is None:
                print(f"No data extension found in {fits_path}")
                continue

            data = hdul[ext].data
            header = hdul[ext].header
            wcs = WCS(header)

        for i, region in enumerate(regions_list, start=1):
            pix_region = region.to_pixel(wcs)
            bbox = pix_region.bounding_box
            x_min, x_max = int(bbox.ixmin), int(bbox.ixmax)
            y_min, y_max = int(bbox.iymin), int(bbox.iymax)

            cropped_data = data[y_min:y_max, x_min:x_max]
            cropped_header = header.copy()
            cropped_header['CRPIX1'] -= x_min
            cropped_header['CRPIX2'] -= y_min

            # Save cut fits
            region_tag = f"reg{i}_" if len(regions_list) > 1 else ""
            output_path = os.path.join(output_dir, f"{region_tag}{filter_name}.fits")

            hdu = fits.PrimaryHDU(data=cropped_data, header=cropped_header)
            hdu.writeto(output_path, overwrite=True)
            print(f"Saved: {output_path}")


# --------- # --------- #

def visualize_fits(fits_path, save_path=None, stretch='log', min_percent=25., max_percent=99.98,
                   cmap='viridis', xlim=None, ylim=None):
    """
    Visualize a fits file with both pixel and RA/Dec axes.

    Parameters
    ----------
    fits_path : str
        Path to the fits file.
    save_path : str
        Path to save the output image. If "None" the figure is just showed, but not saved.
    stretch : str
        Stretch type for simple_norm (e.g., 'linear', 'log', 'sqrt').
    min_percent, max_percent : float
        Percentile limits for normalization.
    cmap : str
        Colormap.
    xlim, ylim : tuple or None
        Pixel axis limits.
    """
    # Open fits
    with fits.open(fits_path, memmap=False) as hdul:
        ext = find_data_extension(hdul)
        if ext is None:
            print(f"No data extension found in {fits_path}")
            return

        data = hdul[ext].data
        header = hdul[ext].header
        wcs = WCS(header)

    # Mask for normalization
    mask = np.logical_or(np.isnan(data), data <= 0.)
    norm = simple_norm(data[~mask], stretch=stretch,
                       min_percent=min_percent, max_percent=max_percent)

    # Figure
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': wcs})
    im = ax.imshow(data, cmap=cmap, origin='lower', norm=norm, interpolation='nearest')

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('x (pixel)')
    secay = ax.secondary_yaxis('right')
    secay.set_ylabel('y (pixel)')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.grid()
    
    base = os.path.basename(fits_path)
    region = base.split('_')[0]  
    ax.set_title(f'{region}')

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f'Save image at {save_path}')
    else:
        plt.show()
        
# to do: a function to visualize each filter stacked in a datacube.    


    
# --------- # --------- #
    
def align_fits(fits_files, reference_file):
    """
    Aligns fits files to a common WCS based on a chosen reference filter and saves the files.

    Parameters
    ----------
    fits_files : list of tuples
        List of fits files in the format [(filename, filter_name), ...] (e.g., [('F070W.fits', 'F070W'), ('F090W.fits', 'F090W'), ...]).
    reference_filter : str
        Path of the filter to use as the alignment reference.

    Returns
    -------
    aligned_images : list of np.ndarray
        List of aligned image arrays.
    filter_names : list of str
        List of filter names corresponding to each image.
    """
    # Open reference filter
    with fits.open(reference_file) as hdul_ref:
        ext_ref = find_data_extension(hdul_ref)
        if ext_ref is None:
            raise ValueError(f"No valid image data in reference file '{reference_fits}'.")
        ref_header = hdul_ref[ext_ref].header

    aligned_filenames = []

    # Align and save fits
    for file, filt in fits_files:
        with fits.open(file) as hdul:
            ext = find_data_extension(hdul)
            if ext is None:
                print(f'No data extension found in {file}. Skipping.')
                continue

            data = hdul[ext].data
            wcs_in = WCS(hdul[ext].header, hdul)

            data_aligned, _ = reproject_interp((data, wcs_in), ref_header)

            aligned_header = ref_header.copy()
            aligned_header.update(WCS(ref_header).to_header())

            aligned_name = os.path.splitext(file)[0] + '_aligned.fits'
            fits.PrimaryHDU(data_aligned, header=aligned_header).writeto(aligned_name, overwrite=True)

            aligned_filenames.append((aligned_name, filt))
            print(f"Saved at '{aligned_name}'")

    return aligned_filenames
    
    
# --------- # --------- #
    
def build_datacube(aligned_fits_files, reference_file, output_path):
    """
    Builds and saves a 3D fits datacube from aligned fits images.

    Parameters
    ----------
    aligned_fits_files : list of tuples
        List of aligned fits files in the format [(filename, filter_name), ...] (e.g., [('F070W_aligned.fits', 'F070W'), ('F090W_aligned.fits', 'F090W'), ...]).
    reference_file : str
        Path to the fits file to use as WCS reference.
    output_path : str, optional
        Path to the output fits file.

    Returns
    -------
    None
        Saves the datacube.
    """
    # Get WCS and shape from reference file
    with fits.open(reference_file) as hdul_ref:
        ext_ref = find_data_extension(hdul_ref)
        if ext_ref is None:
            raise ValueError(f"No valid image data in reference file '{reference_file}'.")
        ref_header = hdul_ref[ext_ref].header
        ny, nx = hdul_ref[ext_ref].data.shape

    aligned_images = []
    filter_names = []
    units = []

    # Get aligned fits, filter names, units
    for file, filt in aligned_fits_files:
        with fits.open(file) as hdul:
            ext = next((i for i, h in enumerate(hdul) if h.data is not None and isinstance(h.data, np.ndarray)), None)
            if ext is None:
                print(f"No data extension found in '{file}'.")
                continue
            data = hdul[ext].data
            
            unit = hdul[ext].header.get('BUNIT', 'unknown')  
            print(f"Filter {filt}: unity = {unit}")  # you can comment this line, it's just a verification
        
            aligned_images.append(data)
            filter_names.append(filt)
            units.append(unit)

    # Stack images into 3D cube
    cube = np.array(aligned_images)

    # Build 3D WCS
    wcs_3d = WCS(naxis=3)
    wcs_2d = WCS(ref_header, naxis=2)

    # Copy spatial info
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

    # Filter axis
    wcs_3d.wcs.crpix[2] = 1
    wcs_3d.wcs.crval[2] = 0
    wcs_3d.wcs.cdelt[2] = 1
    wcs_3d.wcs.ctype[2] = 'FILTER'
    wcs_3d.wcs.cunit[2] = ''

    # Build header
    cube_header = wcs_3d.to_header()
    
    cube_header['NAXIS'] = 3
    cube_header['NAXIS1'] = nx
    cube_header['NAXIS2'] = ny
    cube_header['NAXIS3'] = len(filter_names)
    
    if units: # this part is working but after saving the datacube if you do print(hdul[0].header.get('BUNIT')), it returns "None" :(
        if all(u == units[0] for u in units):
            cube_header['BUNIT'] = units[0]
        else:
            print('Different units', units)
            cube_header['BUNIT'] = 'unknown'
    else:
        print('No BUNIT found in original fits.')
        cube_header['BUNIT'] = 'unknown'

    for i, filt in enumerate(filter_names):
        cube_header[f'FILTER{i+1}'] = filt  # the name of the filter doesn't appear when you open the datacube in DS9... this information is just added in the header of the files :(

    # Save datacube
    fits.PrimaryHDU(cube, header=cube_header).writeto(output_path, overwrite=True)
    print(f"Datacube saved at '{output_path}'")
    

# --------- # --------- #

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
        
       
