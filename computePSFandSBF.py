import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.table import Table
from astropy.io import fits, ascii
import traceback

from euclid_sbf_pipeline import MainPipeline
from psfextraction import extractPSF
from colorextraction import computeColor

import sys
sys.path.append("./functions")

# Own function imports 
from plotting import imdisplay

# output_directory = # maybe set this
gals = sorted(os.listdir('output'))

# comp_gals = Table(names=('OBJECT_ID', 'sbfVIS', 'sbfH', 'sbfmagVIS', 'sbfmagH', '(I-H)', 'backgrVIS', 'backgrH', 'smaVIS', 'smaH', 'mzpVIS', 'mzpH','commentVIS','commentH')
#                 , dtype=(str, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, str, str))

comp_gals = ascii.read("comp_gals.csv")

for gal in gals:
    # initialize all fields as None in case something fails
    # row = {
    #     'OBJECT_ID': gal,
    #     'sbfVIS': None, 'sbfH': None,
    #     'sbfmagVIS': None, 'sbfmagH': None,
    #     '(I-H)':None,
    #     'backgrVIS': None, 'backgrH': None,
    #     'smaVIS': None, 'smaH': None,
    #     'mzpVIS': None, 'mzpH': None,
    #     'commentVIS': '-', 'commentH': '-'
    # }
    # for filter in ['VIS','H']:
    #     try:
    #         # compute the psf
    #         field_path = 'output/{}/{}/psf/field.fits'.format(gal,filter)
    #         return_path = 'output/{}/{}/psf'.format(gal,filter)
    #         extractPSF(field_path, return_path, filter, image_type='stacked')

    #         # compute the sbf
    #         psf_path = 'output/{}/{}/psf/stars.psf'.format(gal,filter)
    #         psf_data = fits.getdata(psf_path, ext=1,header=False)
    #         psf = psf_data[0][0][0]
    #         psf /= np.sum(psf)

    #         data_path = 'output/{}/{}/gal'.format(gal,filter)
    #         file_path = 'output/{}/{}/return_files'.format(gal,filter)
    #         image_path = 'output/{}/{}/images'.format(gal,filter)
    #         sbf, sbfmag, total_bckgr, sma, mzp = MainPipeline(data_path, file_path=file_path, image_path=image_path, make_plots=True, plot_plots=False, psf=psf, maxarea_sourcemask=None, filter=filter, background_estimation=True, background=None, ellipsefitter='v6')
            
    #         # Save results into temporary structure
    #         row[f'sbf{filter}']    = sbf
    #         row[f'sbfmag{filter}'] = sbfmag
    #         row[f'backgr{filter}'] = total_bckgr
    #         row[f'sma{filter}']    = sma
    #         row[f'sma{filter}']    = sma
    #         row[f'mzp{filter}']    = mzp

    #     except Exception as e:
    #         # If something goes wrong, keep None and print what failed
    #         print(f"Warning: {gal}, filter {filter} failed with error:")
    #         print(e)
    #         traceback.print_exc()
    #         row[f'comment{filter}']  = str(e)
    #         pass
    try:
        # compute the color 
        pathVIS = 'output/{}/VIS/return_files'.format(gal)
        pathH = 'output/{}/H/return_files'.format(gal)
        mask = np.loadtxt(pathVIS + '/combined_final_mask')
        dataVIS = np.loadtxt(pathVIS + '/data_background_subtracted')
        dataH = np.loadtxt(pathH + '/data_background_subtracted')
        dataH = np.flip(dataH, axis=0)

        mzpVIS = comp_gals['mzpVIS'] [comp_gals['OBJECT_ID']==gal][0] #row['mzpVIS']
        mzpH = comp_gals['mzpH'][comp_gals['OBJECT_ID']==gal][0]  #row['mzpH']
        print(mzpVIS,mzpH)
        color,dataBmasked,dataRmasked = computeColor(dataVIS, dataH, mask, mzpVIS, mzpH, do_print=True)
        # row['(I-H)'] = color
        comp_gals['(I-H)'][comp_gals['OBJECT_ID']==gal] = color
        for filter,datamasked in zip(['VIS','H'],[dataBmasked,dataRmasked]):
            fig, ax = plt.subplots(figsize=(8, 8))
            imdisplay(datamasked, ax, percentlow=1, percenthigh=99, scale='asinh')
            plt.title("Flux_data")
            image_title = "10.1_flux_data.png"
            image_path = 'output/{}/{}/images'.format(gal,filter)
            plt.savefig(image_path + "/" + image_title)

    except:
        traceback.print_exc()
        pass

    # After both VIS and H are processed - append one row
    # comp_gals.add_row([
    #     row['OBJECT_ID'],
    #     row['sbfVIS'], row['sbfH'],
    #     row['sbfmagVIS'], row['sbfmagH'],
    #     row['(I-H)'],
    #     row['backgrVIS'], row['backgrH'],
    #     row['smaVIS'], row['smaH'],
    #     row['mzpVIS'], row['mzpH'],
    #     row['commentVIS'], row['commentH']
    # ])

comp_gals.write("comp_gals_copy.csv", format="csv", overwrite=True)
