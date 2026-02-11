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

output_folder = 'output_Fornax_sota_v6'

dwarfs_cat = pd.read_csv("input_data/HabasFornax.csv", sep=';')
# dwarfs_cat = Table.from_pandas(dwarfs_cat)
# print(dwarfs_cat.columns)
table_flagged = pd.read_csv('Fornax_flag_table.txt', sep=' ')
# table_flagged =  Table.from_pandas(table_flagged)

hdulI = fits.open('input_data/Euclid-VIS-ERO-Fornax-LSB.DR3.fits')
full_dataI = hdulI[0].data
full_headerI = hdulI[0].header
wcsI = WCS(full_headerI)
hdulI.close()

hdulH = fits.open('input_data/Euclid-NISP-H-ERO-Fornax-LSB.DR3.fits')
full_dataH = hdulH[0].data
full_headerH = hdulH[0].header
wcsH = WCS(full_headerH)
hdulH.close()

for o_id in dwarfs_cat['Name']:
    name = str(o_id).replace(' ','')#.removeprefix('-')
    flag = table_flagged.loc[table_flagged['OBJECT_ID'] == name, 'flag'].iloc[0]
    # print(flag)
    if flag == 0:
        cutout_size = round(dwarfs_cat.loc[dwarfs_cat['Name']==o_id,'Re'].iloc[0]*30)
        # print(cutout_size)
        ra_center = dwarfs_cat.loc[dwarfs_cat['Name']==o_id,'RA'].iloc[0]
        dec_center = dwarfs_cat.loc[dwarfs_cat['Name']==o_id,'Dec'].iloc[0]
        position = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg), frame='icrs')
    
        try:
            # cutout for the VIS band
            cutoutI = Cutout2D(full_dataI, position, (cutout_size*3, cutout_size*3), wcs=wcsI, copy=True)
            new_headerI = full_headerI.copy()
            new_headerI.update(cutoutI.wcs.to_header())
            hdu_cutoutI = fits.PrimaryHDU(data=cutoutI.data, header=new_headerI)
            try:
                os.makedirs("{}/{}/VIS/gal".format(output_folder, name))
            except FileExistsError:
                pass
            hdu_cutoutI.writeto("{}/{}/VIS/gal/{}.fits".format(output_folder, name, name), overwrite=True)
    
            fieldcutoutI = Cutout2D(full_dataI, position, (cutout_size*6, cutout_size*6), wcs=wcsI, copy=True)
            new_header_fieldI = full_headerI.copy()
            new_header_fieldI.update(fieldcutoutI.wcs.to_header())
    
            hdu_fieldcutoutI = fits.PrimaryHDU(data=fieldcutoutI.data, header=new_header_fieldI)
            hdu_fieldcutoutI.data[cutout_size*3//2:cutout_size*9//2, cutout_size*3//2:cutout_size*9//2] = np.nan
            try:
                os.makedirs("{}/{}/VIS/psf".format(output_folder,name))
            except FileExistsError:
                pass
            hdu_fieldcutoutI.writeto("{}/{}/VIS/psf/field.fits".format(output_folder,name), overwrite=True)
    
            # cutout for the H-band
            cutoutH = Cutout2D(full_dataH, position, (cutout_size, cutout_size), wcs=wcsH, copy=True)
            new_headerH = full_headerH.copy()
            new_headerH.update(cutoutH.wcs.to_header())
            hdu_cutoutH = fits.PrimaryHDU(data=cutoutH.data, header=new_headerH)
            try:
                os.makedirs("{}/{}/H/gal".format(output_folder,name))
            except FileExistsError:
                pass
            hdu_cutoutH.writeto("{}/{}/H/gal/{}.fits".format(output_folder,name, name), overwrite=True)
    
            fieldcutoutH = Cutout2D(full_dataH, position, (cutout_size*2, cutout_size*2), wcs=wcsH, copy=True)
            new_header_fieldH = full_headerH.copy()
            new_header_fieldH.update(fieldcutoutH.wcs.to_header())
    
            hdu_fieldcutoutH = fits.PrimaryHDU(data=fieldcutoutH.data, header=new_header_fieldH)
            hdu_fieldcutoutH.data[cutout_size//2:cutout_size*3//2, cutout_size//2:cutout_size*3//2] = np.nan
            try:
                os.makedirs("{}/{}/H/psf".format(output_folder,name))
            except FileExistsError:
                pass
            hdu_fieldcutoutH.writeto("{}/{}/H/psf/field.fits".format(output_folder,name), overwrite=True)
        except:
            continue
    else:
        continue
