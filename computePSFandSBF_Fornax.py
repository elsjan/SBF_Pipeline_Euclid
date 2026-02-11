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

output_folder = "output_Fornax_sota_v6"
gals = sorted(os.listdir(output_folder))

comp_gals = Table(names=('OBJECT_ID', 'sbfVIS', 'sbfH', 'sbfmagVIS', 'sbfmagH', '(I-H)', 'backgrVIS', 'backgrH', 'smaVIS', 'smaH','commentVIS','commentH')
                , dtype=(str, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, str, str))

# comp_gals = ascii.read("comp_gals.csv")

for gal in gals:
    # initialize all fields as None in case something fails
    row = {
        'OBJECT_ID': gal,
        'sbfVIS': None, 'sbfH': None,
        'sbfmagVIS': None, 'sbfmagH': None,
        '(I-H)':None,
        'backgrVIS': None, 'backgrH': None,
        'smaVIS': None, 'smaH': None,
        'commentVIS': '-', 'commentH': '-'
    }
    for filter in ['VIS','H']:
        try:
            # compute the psf
            field_path = '{}/{}/{}/psf/field.fits'.format(output_folder,gal,filter)
            return_path = '{}/{}/{}/psf'.format(output_folder,gal,filter)
            extractPSF(field_path, return_path, filter)

            # compute the sbf
            psf_path = '{}/{}/{}/psf/stars.psf'.format(output_folder,gal,filter)
            psf_data = fits.getdata(psf_path, ext=1,header=False)
            psf = psf_data[0][0][0]
            psf /= np.sum(psf)

            data_path = '{}/{}/{}/gal'.format(output_folder,gal,filter)
            file_path = '{}/{}/{}/return_files'.format(output_folder,gal,filter)
            image_path = '{}/{}/{}/images'.format(output_folder,gal,filter)
            sbf, sbfmag, total_bckgr, sma, mzp = MainPipeline(data_path, file_path=file_path, field_path=field_path, image_path=image_path, make_plots=True, plot_plots=False, psf=psf, maxarea_sourcemask=None, filter=filter, background_estimation=True, background=None, ellipsefitter='v6', sma_rescale=0.8)
            
            # Save results into temporary structure
            row[f'sbf{filter}']    = sbf
            row[f'sbfmag{filter}'] = sbfmag
            row[f'backgr{filter}'] = total_bckgr
            row[f'sma{filter}']    = sma

        except Exception as e:
            # If something goes wrong, keep None and print what failed
            print(f"Warning: {gal}, filter {filter} failed with error:")
            print(e)
            traceback.print_exc()
            row[f'comment{filter}']  = str(e)
            pass
    try:
        # compute the color 
        pathVIS = '{}/{}/VIS/return_files'.format(output_folder,gal)
        pathH = '{}/{}/H/return_files'.format(output_folder,gal)
        mask = np.loadtxt(pathVIS + '/color_final_mask')
        dataVIS = np.loadtxt(pathVIS + '/data_background_subtracted')
        dataH = np.loadtxt(pathH + '/data_background_subtracted')
#         dataH = np.flip(dataH, axis=0)

        mzpVIS = np.loadtxt(pathVIS + '/background')[2]
        mzpH =  np.loadtxt(pathH + '/background')[2]
        print('mzp:',mzpVIS,mzpH)

        color,dataBmasked,dataRmasked = computeColor(dataVIS, dataH, mask, mzpVIS, mzpH, do_print=True)
        row['(I-H)'] = color
        for filter,datamasked in zip(['VIS','H'],[dataBmasked,dataRmasked]):
            fig, ax = plt.subplots(figsize=(8, 8))
            imdisplay(datamasked, ax, percentlow=1, percenthigh=99, scale='asinh')
            plt.title("Flux_data")
            image_title = "10.1_flux_data.png"
            image_path = '{}/{}/{}/images'.format(output_folder,gal,filter)
            plt.savefig(image_path + "/" + image_title)

    except:
        traceback.print_exc()
        pass

#     After both VIS and H are processed - append one row
    comp_gals.add_row([
        row['OBJECT_ID'],
        row['sbfVIS'], row['sbfH'],
        row['sbfmagVIS'], row['sbfmagH'],
        row['(I-H)'],
        row['backgrVIS'], row['backgrH'],
        row['smaVIS'], row['smaH'],
        row['commentVIS'], row['commentH']
    ])

comp_gals.write("comp_gals_Fornax_sota_v6.csv", format="csv", overwrite=True)
