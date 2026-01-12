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

dwarfs_cat = pd.read_csv("input_data/dorado_t21_t24_mer.csv")
dwarfs_cat.sort_values(by=['mag']) # so we get the brightest galaxies first
dwarfs_cat = Table.from_pandas(dwarfs_cat)

hdulI = fits.open('input_data/Euclid-VIS-ERO-Dorado-LSB.v3.fits')
full_dataI = hdulI[0].data
full_headerI = hdulI[0].header
wcsI = WCS(full_headerI)
hdulI.close()

hdulH = fits.open('input_data/Euclid-NISP-H-ERO-Dorado-LSB.v3.fits')
full_dataH = hdulH[0].data
full_headerH = hdulH[0].header
wcsH = WCS(full_headerH)
hdulH.close()

for o_id in dwarfs_cat['OBJECT_ID']:
    # only the red galaxies right now
    if dwarfs_cat[dwarfs_cat['OBJECT_ID']==o_id]['ColorID'] == 'r':
        name = str(o_id).replace('-','')#.removeprefix('-')
        cutout_size = round(dwarfs_cat[dwarfs_cat['OBJECT_ID']==o_id]['Re_arcsec'].value[0]*30)
        ra_center = dwarfs_cat[dwarfs_cat['OBJECT_ID']==o_id]['RIGHT_ASCENSION']
        dec_center = dwarfs_cat[dwarfs_cat['OBJECT_ID']==o_id]['DECLINATION']
        position = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg), frame='icrs')

        # seems like only a few galaxies are actually positioned in the field image i have
        try:
            # cutout for the VIS band
            cutoutI = Cutout2D(full_dataI, position, (cutout_size*3, cutout_size*3), wcs=wcsI, copy=True)
            new_headerI = full_headerI.copy()
            new_headerI.update(cutoutI.wcs.to_header())
            hdu_cutoutI = fits.PrimaryHDU(data=cutoutI.data, header=new_headerI)
            try:
                os.makedirs("output/{}/VIS/gal".format(name))
            except FileExistsError:
                pass
            hdu_cutoutI.writeto("output/{}/VIS/gal/{}.fits".format(name, name), overwrite=True)

            fieldcutoutI = Cutout2D(full_dataI, position, (cutout_size*6, cutout_size*6), wcs=wcsI, copy=True)
            new_header_fieldI = full_headerI.copy()
            new_header_fieldI.update(fieldcutoutI.wcs.to_header())

            hdu_fieldcutoutI = fits.PrimaryHDU(data=fieldcutoutI.data, header=new_header_fieldI)
            hdu_fieldcutoutI.data[cutout_size*3//2:cutout_size*9//2, cutout_size*3//2:cutout_size*9//2] = np.nan
            try:
                os.makedirs("output/{}/VIS/psf".format(name))
            except FileExistsError:
                pass
            hdu_fieldcutoutI.writeto("output/{}/VIS/psf/field.fits".format(name), overwrite=True)

            # cutout for the H-band
            cutoutH = Cutout2D(full_dataH, position, (cutout_size, cutout_size), wcs=wcsH, copy=True)
            new_headerH = full_headerH.copy()
            new_headerH.update(cutoutH.wcs.to_header())
            hdu_cutoutH = fits.PrimaryHDU(data=cutoutH.data, header=new_headerH)
            try:
                os.makedirs("output/{}/H/gal".format(name))
            except FileExistsError:
                pass
            hdu_cutoutH.writeto("output/{}/H/gal/{}.fits".format(name, name), overwrite=True)
            
            fieldcutoutH = Cutout2D(full_dataH, position, (cutout_size*2, cutout_size*2), wcs=wcsH, copy=True)
            new_header_fieldH = full_headerH.copy()
            new_header_fieldH.update(fieldcutoutH.wcs.to_header())

            hdu_fieldcutoutH = fits.PrimaryHDU(data=fieldcutoutH.data, header=new_header_fieldH)
            hdu_fieldcutoutH.data[cutout_size//2:cutout_size*3//2, cutout_size//2:cutout_size*3//2] = np.nan
            try:
                os.makedirs("output/{}/H/psf".format(name))
            except FileExistsError:
                pass
            hdu_fieldcutoutH.writeto("output/{}/H/psf/field.fits".format(name), overwrite=True)
        except:
            continue

