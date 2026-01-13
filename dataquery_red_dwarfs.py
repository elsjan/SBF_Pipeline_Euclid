import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS 
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy import units as u

# THIS CODE IS TOO COMPLEX FOR THE NEEDS, USE ORIG DWARFS DATAQUERY INSTEAD 

dwarfs_cat = pd.read_csv("input_data/dorado_t21_t24_mer.csv")
dwarfs_cat['OBJECT_ID'] = dwarfs_cat['OBJECT_ID'].astype(str)
dwarfs_cat['cutout_ids'] = dwarfs_cat['ra'].astype(str)+'_'+dwarfs_cat['dec'].astype(str)
# dwarfs_cat.sort_values(by=['mag']) # so we get the brightest galaxies first
# dwarfs_cat = Table.from_pandas(dwarfs_cat)

VIS_files_path = "input_data/cutoutsVIS"
H_files_path =  "input_data/cutoutsH"
output_folder = "output"

gals_vis = sorted(os.listdir(VIS_files_path))
gals_h = sorted(os.listdir(H_files_path))

Re_scale = 30


for gal in gals_vis:
    try:
        # cutout for the VIS band
        name = str(gal).replace('.fits', '') #maybe .fits shows up and should be removed
        name = name.split("CUTOUT_", 1)[1]

        hdulI = fits.open(f'{VIS_files_path}/{gal}')
        full_dataI = hdulI[0].data
        w,l = len(full_dataI[0,:]),len(full_dataI[:,0])
        if w != l:
            if w > l:
                diff = int(w-l)
                full_dataI = full_dataI[:,:-diff]
            elif l > w:
                diff = int(l-w)
                full_dataI = full_dataI[:-diff,:]

        full_headerI = hdulI[0].header
        full_headerI['ZP_STACK'] = float(full_headerI['MAGZEROP'])
        wcsI = WCS(full_headerI)
        
        s = dwarfs_cat[dwarfs_cat['cutout_ids'] ==name]['Re_arcsec']
        j = s.values[0]*30*3
        Re_arcsec = int(j)
        border_size = (len(full_dataI) - Re_arcsec)//2
        cutout_size = 150*3
        cutoutI = full_dataI[border_size:-border_size,border_size:-border_size]
        new_headerI = full_headerI.copy()
        # new_headerI.update(cutoutI.wcs.to_header()) 
        hdu_cutoutI = fits.PrimaryHDU(data=cutoutI, header=new_headerI)
        try:
            os.makedirs("{}/{}/VIS/gal".format(output_folder, name))
        except FileExistsError:
            pass
        hdu_cutoutI.writeto("{}/{}/VIS/gal/{}.fits".format(output_folder, name, name), overwrite=True)

        fieldcutoutI = full_dataI.copy()
        fieldcutoutI[cutout_size:-cutout_size, cutout_size:-cutout_size] = np.nan
        hdu_fieldcutoutI = fits.PrimaryHDU(data=fieldcutoutI, header=new_headerI)
        try:
            os.makedirs("{}/{}/VIS/psf".format(output_folder,name))
        except FileExistsError:
            pass
        hdu_fieldcutoutI.writeto("{}/{}/VIS/psf/field.fits".format(output_folder,name), overwrite=True)

        hdulI.close()
    except Exception as e:
        print(e)
        continue

for gal in gals_h:
    try:
        # cutout for the H-band
        name = str(gal).replace('.fits', '') #maybe .fits shows up and should be removed
        name = name.split("CUTOUT_", 1)[1]
        # obj_id = '-'+name
        hdulH = fits.open(f'{H_files_path}/{gal}')
        full_dataH = hdulH[0].data
        w,l = len(full_dataH[0,:]),len(full_dataH[:,0])
        if w != l:
            if w > l:
                diff = int(w-l)
                full_dataH = full_dataH[:,:-diff]
            elif l > w:
                diff = int(l-w)
                full_dataH = full_dataH[:-diff,:]

        full_headerH = hdulH[0].header
        full_headerH['ZP_STACK'] = full_headerH['ZPAB'] #30.0 GAIN SATURATE
        wcsH = WCS(full_headerH)

        s = dwarfs_cat[dwarfs_cat['cutout_ids'] ==name]['Re_arcsec']
        j = s.values[0]*30
        Re_arcsec = int(j)
        border_size = (len(full_dataH) - Re_arcsec)//2
        cutout_size = 150
        cutoutH = full_dataH[border_size:-border_size,border_size:-border_size]
        new_headerH = full_headerH.copy()
        # new_headerH.update(cutoutH.wcs.to_header())
        hdu_cutoutH = fits.PrimaryHDU(data=cutoutH, header=new_headerH)
        try:
            os.makedirs("{}/{}/H/gal".format(output_folder,name))
        except FileExistsError:
            pass
        hdu_cutoutH.writeto("{}/{}/H/gal/{}.fits".format(output_folder,name, name), overwrite=True)
        
        fieldcutoutH = full_dataH.copy()
        fieldcutoutH[cutout_size:-cutout_size, cutout_size:-cutout_size] = np.nan
        hdu_fieldcutoutH = fits.PrimaryHDU(data=fieldcutoutH, header=new_headerH)
        try:
            os.makedirs("{}/{}/H/psf".format(output_folder,name))
        except FileExistsError:
            pass
        hdu_fieldcutoutH.writeto("{}/{}/H/psf/field.fits".format(output_folder,name), overwrite=True)
        hdulH.close()

    except Exception as e:
        print(e)
        continue

