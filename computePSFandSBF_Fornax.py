import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.table import Table
from astropy.io import fits, ascii
import traceback
import time

from euclid_sbf_pipeline import MainPipeline
from psfextraction import extractPSF
from colorextraction import computeColor, computeColorPSFconv, computeColorCombinedMask

import sys
sys.path.append("./functions")

# Own function imports 
from plotting import imdisplay


run = 'Fornax_10'


output_folder = f'output_{run}' 
gals = sorted(os.listdir(output_folder))

comp_gals = Table(names=('OBJECT_ID', 'sbfVIS', 'sbferrVIS', 'sbfH', 'sbferrH', 'sbfmagVIS', 'sbfmagH', 'noiseVIS', 'noiseH', 
                        'sbfcorVIS', 'sbfcorerrVIS', 'sbfmagcorVIS', 'sbfcorH', 'sbfcorerrH', 'sbfmagcorH', 
                        '(I-H)', 'backgrVIS', 'backgrH', 'smaVIS', 'smaH','npixVIS','npixH','commentVIS','commentH')
                , dtype=(str, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, 
                        np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, 
                        np.float64, np.float64, np.float64, np.float64,int,int, str, str))

# comp_gals = ascii.read("comp_gals.csv")

dwarfs_cat = pd.read_csv("input_data/HabasFornax.csv", sep=',')

# psf_path = 'psf_Fornax_LSB/VIS/psfs/'
# # psf_names = os.listdir(psf_path)
# # psf_names = np.array([s for s in psf_names if "psf" in s])
# psf_names = np.array((0,1,2,3,5,16,17,19,20,22,26,28,30,33,35,41,42,46,50,53,57,58,65,77,81,82,83,85,86,88,90,94)).astype(str)

# psfsVIS = np.zeros((len(psf_names),35,35))

# i=0
# for psf_name in psf_names:
#     hdul = fits.open(f"{psf_path}psf{psf_name}.fits")
#     psf = hdul[0].data
#     hdul.close()
#     psf /= np.sum(psf)
#     psfsVIS[i,:,:] = psf
#     i+=1

# psf_path = 'psf_Fornax_LSB/H/psfs/'
# # psf_names = os.listdir(psf_path)
# # psf_names = np.array([s for s in psf_names if "psf" in s])
# psf_names = np.array((4,5,6,8,18,25,27,28,29,35,37,39,41,47,48,55,60,61,68,69,74,75,81,96,103,110,112)).astype(str)

# psfsH = np.zeros((len(psf_names),35,35))

# i=0
# for psf_name in psf_names:
#     hdul = fits.open(f"{psf_path}psf{psf_name}.fits")
#     psf = hdul[0].data
#     hdul.close()
#     psf /= np.sum(psf)
#     psfsH[i,:,:] = psf
#     i+=1

# np.save('psf_Fornax_LSB/VIS/psfsFornaxLSBVIS', psfsVIS)
# np.save('psf_Fornax_LSB/H/psfsFornaxLSBH', psfsH)

psfsVIS = np.load('psf_Fornax_LSB/VIS/psfsFornaxLSBVIS.npy')
psfsH = np.load('psf_Fornax_LSB/H/psfsFornaxLSBH.npy')


time_elaps = pd.DataFrame(data=None, columns=['Name', 'time'], dtype=(str,np.float64))
for gal in gals:
    # initialize all fields as None in case something fails
    row = {
        'OBJECT_ID': gal,
        'sbfVIS': None, 'sbferrVIS': None,
        'sbfH': None, 'sbferrH': None,
        'sbfmagVIS': None, 'sbfmagH': None,
        'noiseVIS': None, 'noiseH': None,
        'sbfcorVIS': None, 'sbfcorerrVIS': None, 'sbfmagcorVIS': None, 
        'sbfcorH': None, 'sbfcorerrH': None, 'sbfmagcorH': None, 
        '(I-H)':None,
        'backgrVIS': None, 'backgrH': None,
        'smaVIS': None, 'smaH': None,
        'npixVIS': 0, 'npixH': 0,
        'commentVIS': '-', 'commentH': '-'
    }
    # print(dwarfs_cat['Name'])
    # print(dwarfs_cat.loc[dwarfs_cat['Name']==gal])
    cat_rinout = dwarfs_cat.loc[dwarfs_cat['Name'] == gal, ['rin', 'rout']].iloc[0]
    cat_rin, cat_rout = cat_rinout
    print("Habas in/out:", cat_rinout)
    for filter, psfs in zip(['VIS','H'],[psfsVIS, psfsH]):
        # compute the psf
        field_path = '{}/{}/{}/psf/field.fits'.format(output_folder,gal,filter)
        # return_path = '{}/{}/{}/psf'.format(output_folder,gal,filter)
        # extractPSF(field_path, return_path, filter)

        # compute the sbf
        # psf_path = '{}/{}/{}/psf/stars.psf'.format(output_folder,gal,filter)
        # psf_data = fits.getdata(psf_path, ext=1,header=False)
        # psf = psf_data[0][0][0]
        # psf /= np.sum(psf)

        data_path = '{}/{}/{}/gal'.format(output_folder,gal,filter)
        file_path = '{}/{}/{}/return_files'.format(output_folder,gal,filter)
        image_path = '{}/{}/{}/images'.format(output_folder,gal,filter)
        fits_path = '{}/{}/{}/fits_files'.format(output_folder,gal,filter)

        if filter == 'VIS':
            geometry = None
        elif filter == 'H':
            cat_rin = max(int(np.round(cat_rin/3)), 1)
            cat_rout = int(np.round(cat_rout/3))
            if cat_rout < 18:
                row[f'comment{filter}']  = f'rout too small ({cat_rout})'
                cat_rout = 18
            geometry.x0 = int(np.round(geometry.x0/3))
            geometry.y0 = int(np.round(geometry.y0/3))
            geometry.sma = int(np.round(geometry.sma/3))

        try:
            start = time.perf_counter()   
            sbf, std_p0, sbfmag, noise, sbf_cor, sbfcor_err, sbfmag_cor, sma, rout, total_bckgr, mzp, geometry, Npix = MainPipeline(
                                                                    data_path, file_path=file_path, field_path=field_path, image_path=image_path, 
                                                                     make_plots=True, plot_plots=False, psf=psfs, geometry=geometry,
                                                                     maxarea_sourcemask=None, filter=filter, 
                                                                     background_estimation=True, background_estimation_median=False, background=None, ellipsefitter='v7', sma_rescale=0.8,  
                                                                     maskbackgroundsourcesweighted=True, SN_psfitting=None, 
                                                                     manual_nri_mask=[cat_rin,cat_rout], galmolcorr=False, galmolsmooth=False,
                                                                     residual_sources_estimation=True, fits_path=fits_path)


            
            # Save results into temporary structure
            row[f'sbf{filter}']         = sbf
            row[f'sbferr{filter}']      = std_p0
            row[f'sbfmag{filter}']      = sbfmag
            row[f'noise{filter}']       = noise
            row[f'sbfcor{filter}']      = sbf_cor
            row[f'sbfcorerr{filter}']   = sbfcor_err
            row[f'sbfmagcor{filter}']   = sbfmag_cor
            row[f'backgr{filter}']      = total_bckgr
            row[f'sma{filter}']         = sma
            row[f'npix{filter}']        = Npix
            end = time.perf_counter()
            time_elaps.loc[len(time_elaps)] = gal+filter, end-start
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
        maskVIS = np.loadtxt(pathVIS + '/combined_final_mask')
        maskH = np.loadtxt(pathH + '/combined_final_mask')
        dataVIS = np.loadtxt(pathVIS + '/data_background_subtracted')
        dataH = np.loadtxt(pathH + '/data_background_subtracted')
#         dataH = np.flip(dataH, axis=0)

        mzpVIS = np.loadtxt(pathVIS + '/background')[2]
        mzpH =  np.loadtxt(pathH + '/background')[2]
        print('mzp:',mzpVIS,mzpH)

        color,dataBmasked,dataRmasked = computeColorCombinedMask(dataVIS, dataH, maskVIS, maskH, mzpVIS, mzpH, do_print=True)

        row['(I-H)'] = color
        for filter,datamasked in zip(['VIS','H'],[dataBmasked,dataRmasked]):
            x,y = geometry.x0, geometry.y0
            r = rout+2
            fig, ax = plt.subplots(figsize=(8, 8))
            imdisplay(datamasked[y-r:y+r,x-r:x+r], ax, percentlow=1, percenthigh=99, scale='asinh')
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
        row['sbfVIS'], row['sbferrVIS'], 
        row['sbfH'], row['sbferrH'],
        row['sbfmagVIS'], row['sbfmagH'],
        row['noiseVIS'], row['noiseH'],
        row['sbfcorVIS'], row['sbfcorerrVIS'], row['sbfmagcorVIS'], 
        row['sbfcorH'], row['sbfcorerrH'], row['sbfmagcorH'],
        row['(I-H)'],
        row['backgrVIS'], row['backgrH'],
        row['smaVIS'], row['smaH'],
        row['npixVIS'], row['npixH'],
        row['commentVIS'], row['commentH']
    ])

comp_gals.write(f'comp_gals_{run}.csv', format="csv", overwrite=True)
time_elaps.to_csv(f'time_elaps_{run}.txt', index=False)
