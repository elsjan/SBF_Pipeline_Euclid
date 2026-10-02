##########################################################################
# Pipeline for SBF amplitude calculation, specifically for Euclid data, 
# based on the code of Lei Titulaer

# Version 2. Implementing unity in geometry
##########################################################################

version = "3.4"

## 3.4
# added combined mask that determines sbf calculation region into background patch mask

## 3.3
# proper wrapper implementation

# isophote fit, aperture backup
# SN based on RMS and manual selection option

# changes: 
# weighted sourcemask option (maskbackgroundsourcesweighted)
# model correction option (galmolcorr)
# SN based region selection option (SN_psfitting)
# power spectrum fit plotting option (plotyrange)
# fits file saving option (fits_path)

# previous changes
# - improved version aperture model
# - background calculation not disturbed by nans/infs
# - masked objects larger masks
# - ellipse fit smaller step size

# Imports 

import os 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from astropy.io import fits
from astropy.wcs import WCS 
from astroscrappy import detect_cosmics
import csv
from datetime import datetime

from mgefit.find_galaxy import find_galaxy


# Checking current path and adding own functions

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "functions"))

# Own function imports 
from plotting import imdisplay
from extractdata import extractData
from backgroundmodel import mainBackgroundEstimation, sexFieldBackgroundEstimation, sexFieldFromDataBackgroundEstimation
from ellipsemodels import MainFitEllipseModel, fitApertureModel, modelBackgroundSubtraction
from sourcemasking import maskBackgroundSources, maskCircle, createRequiredVariables, maskBackgroundSourcesWeighted
from empiricalpsf import extractPsfSources
from fourierfunctions import calculateSBF, appSBFmagnitude
from librarypsfhubble import calculateLibrarySBF
from sbfuncertainties import sbfMagnitudeAnnuliSigmas


# ---------------------------------


###############
# Main SBF calculation pipeline function
###############


def MainPipeline(data_path, mask_path=None, file_path=None, field_path=None, field_from_data=False, image_path=None, fits_path=None,
                 make_plots=True, plot_plots=True, 
                 psf=None, geometry=None, maxarea_sourcemask=None, filter="VIS", 
                 background_estimation=False, background_estimation_median=False, background=None, ellipsefitter='v7',
                 sma_rescale=0.8, cosmic_ray_method='astroscrappy', premstop=False,
                 plotyrange=None, maskbackgroundsourcesweighted=True,
                 SN_psfitting=None, manual_nri_mask=None,
                 galmolcorr=True, galmolsmooth=False, residual_sources_estimation=True, inv_image_sourcemasking=False):
    """
    Combining the original with the new functions
    """
    print(f"version {version}")
    print(f'computing galaxy from {data_path}')
    with open(f"{file_path}/parameters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["Parameter", "Value"])
        for key, value in locals().items():
            writer.writerow([key, value])
            
    print("\n1. Extracting the data ...")
    plt.close()
    data, mask_cr, wcs, mzp = extractData(data_path, mask_path=mask_path, file_path=file_path, image_path=image_path,fits_path=fits_path,
                                          make_plots=make_plots, plot_plots=plot_plots, 
                                          filter=filter, cosmic_ray_method=cosmic_ray_method)

    if background_estimation:
        if field_from_data:
            print("\n2. Performing sextractor background estimation on field ...")
            data, mask_cr, field_data, total_bckgr, rms_bckgr = sexFieldFromDataBackgroundEstimation(data, mask_cr)  #can specify size of border if wanted
            sex_bckgr = 0
        elif field_path is not None:
            print("\n2. Performing sextractor background estimation on field ...")
            data, field_data, total_bckgr, rms_bckgr = sexFieldBackgroundEstimation(data,field_path)
            sex_bckgr = 0
        elif background_estimation_median:
            print("\n2. Performing background estimation from median image value ...")
            total_bckgr = np.nanmedian(data) 
            data -= total_bckgr
            sex_bckgr = 0
        else:
            print("\n2. Performing background estimation ...")
            data, total_bckgr, sex_bckgr = mainBackgroundEstimation(data, mask_cr, image_path=image_path, make_plots=make_plots, plot_plots=plot_plots)
        if background is not None:
            data += total_bckgr
            total_bckgr = background
            data -= total_bckgr

    else:
        print("\n2. No background estimation performed ...")
        sex_bckgr = 0
        if background is not None:
            total_bckgr = background
            data -= total_bckgr
        else:
            total_bckgr = 0

    if file_path is not None:
        np.savetxt(file_path + "/background", [total_bckgr, sex_bckgr, mzp])
        np.savetxt(file_path + "/data_background_subtracted", data)
    if premstop:
        np.savetxt(file_path + "/mask_initial", ~mask_cr)
        return 0,0,0,0,mzp
    # if maxarea_sourcemask == None:   currently in testing
    #     if filter == 'VIS':
    #         scalar_maxarea = 1
    #     else:
    #         scalar_maxarea = 9
    #     maxarea_sourcemask = 1500/scalar_maxarea
    if ellipsefitter == 'a1':
        print("\n3. Fitting initial aperture ellipse model ...")
        if filter == 'VIS':
            residual_basic, model_basic, geometry = fitApertureModel(data, mask_cr=mask_cr, geometry=None, mgecenter='med',
                                                                    make_plots=make_plots, plot_plots=plot_plots, final=False, image_path=image_path)
        else:
            residual_basic, model_basic, geometry = fitApertureModel(data, mask_cr=mask_cr, geometry=geometry, mgecenter='med',
                                                                    make_plots=make_plots, plot_plots=plot_plots, final=False, image_path=image_path)

        print("\n4. Finding initial source mask ...")
        if maskbackgroundsourcesweighted == True:
            source_mask = maskBackgroundSourcesWeighted(residual_basic, mask_cr=mask_cr, model=model_basic, maxarea=maxarea_sourcemask, r=5, geometry=geometry,
                                                        make_plots=make_plots, plot_plots=plot_plots,  image_path=image_path, final=False, inv_image=inv_image_sourcemasking)
        else:
            source_mask = maskBackgroundSources(residual_basic, mask_cr=mask_cr, maxarea=maxarea_sourcemask, r=5, geometry=geometry,
                                                        make_plots=make_plots, plot_plots=plot_plots,  image_path=image_path, final=False, inv_image=inv_image_sourcemasking)

        print("\n5. Fitting final aperture ellipse model ...")
        if filter == 'VIS':
            residual_final, model_final, geometry = fitApertureModel(data, mask_cr=source_mask, geometry=None, make_plots=make_plots, mgecenter='med',
                                                                    plot_plots=plot_plots, final=True, image_path=image_path)
        else:
            residual_final, model_final, geometry = fitApertureModel(data, mask_cr=source_mask, geometry=geometry, make_plots=make_plots, mgecenter='med',
                                                                    plot_plots=plot_plots, final=True, image_path=image_path)

    else:
        # try:
        #     print("\n3. Fitting initial ellipse model ...")
        #     if filter == 'VIS':
        #         residual_basic, model_basic, geometry, isolist = MainFitEllipseModel(data, mask_cr=mask_cr, geometry=None,
        #                                                                             make_plots=make_plots, plot_plots=plot_plots, 
        #                                                                             image_path=image_path, final=False, method=ellipsefitter, sma_rescale=sma_rescale)
        #     else:
        #         residual_basic, model_basic, geometry, isolist = MainFitEllipseModel(data, mask_cr=mask_cr, geometry=geometry,
        #                                                                             make_plots=make_plots, plot_plots=plot_plots, 
        #                                                                             image_path=image_path, final=False, method=ellipsefitter, sma_rescale=sma_rescale)


        # except:
        print("\n3. Fitting initial aperture ellipse model ...")
        if filter == 'VIS':
            residual_basic, model_basic, geometry = fitApertureModel(data, mask_cr=mask_cr, geometry=None,mgecenter='med',
                                                                    make_plots=make_plots, plot_plots=plot_plots, 
                                                                    final=False, image_path=image_path)
        else:
            residual_basic, model_basic, geometry = fitApertureModel(data, mask_cr=mask_cr, geometry=geometry,mgecenter='med',
                                                    make_plots=make_plots, plot_plots=plot_plots, 
                                                    final=False, image_path=image_path)

        print("\n4. Finding initial source mask ...")
        if maskbackgroundsourcesweighted == True:
            source_mask = maskBackgroundSourcesWeighted(residual_basic, mask_cr=mask_cr, model=model_basic, maxarea=maxarea_sourcemask, r=5, geometry=geometry,
                                                        make_plots=make_plots, plot_plots=plot_plots,  image_path=image_path, final=False, inv_image=inv_image_sourcemasking)
        else:
            source_mask = maskBackgroundSources(residual_basic, mask_cr=mask_cr, maxarea=maxarea_sourcemask, r=5, geometry=geometry,
                                                        make_plots=make_plots, plot_plots=plot_plots,  image_path=image_path, final=False, inv_image=inv_image_sourcemasking)

        try:
            print("\n5. Fitting final ellipse model ...")
            if filter == 'VIS':
                residual_final, model_final, geometry, isolist = MainFitEllipseModel(data, mask_cr=source_mask, geometry=None, mgecenter='med',
                                                                            make_plots=make_plots, plot_plots=plot_plots, final=True, image_path=image_path,
                                                                            method=ellipsefitter, sma_rescale=sma_rescale)
            else:
                residual_final, model_final, geometry, isolist = MainFitEllipseModel(data, mask_cr=source_mask, geometry=geometry, mgecenter='med',
                                                                            make_plots=make_plots, plot_plots=plot_plots, final=True, image_path=image_path,
                                                                            method=ellipsefitter, sma_rescale=sma_rescale)
        except:
            print("\n5. Fitting final aperture ellipse model ...")
            if filter == 'VIS':
                residual_final, model_final, geometry = fitApertureModel(data, mask_cr=source_mask, geometry=None, mgecenter='med',
                                                                        make_plots=make_plots, plot_plots=plot_plots, final=True, image_path=image_path)
            else:
                residual_final, model_final, geometry = fitApertureModel(data, mask_cr=source_mask, geometry=geometry, mgecenter='med',
                                                                        make_plots=make_plots, plot_plots=plot_plots, final=True, image_path=image_path)
    if galmolcorr == True:
        print("\n5.3 Galaxy model correction ... ")
        model_pre_corr = model_final.copy()
        residual_pre_corr = residual_final.copy()
        model_final = modelBackgroundSubtraction(data, model=model_final, mask=mask_cr, final=True, image_path=image_path, 
                                                make_plots=make_plots, plot_plots=plot_plots, bwh=25)
        residual_final = data - model_final
    if galmolsmooth == True:
        from scipy.signal import convolve
        sigma=2
        x = np.linspace(-10, 10, 51)
        y = np.linspace(-10, 10, 51)
        xx, yy = np.meshgrid(x, y)
        gaussian_psf = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
        gaussian_psf /= gaussian_psf.sum()
        model_final = convolve(model_final, gaussian_psf, mode='same', method='auto')
        residual_final = data - model_final
        
    print("\n6. Finding final source mask ...")
    if maskbackgroundsourcesweighted == True:
        source_mask_final = maskBackgroundSourcesWeighted(residual_final, mask_cr=mask_cr, model=model_final, make_plots=make_plots, 
                                                          plot_plots=plot_plots, maxarea=maxarea_sourcemask, image_path=image_path, final=True, 
                                                          r=5, inv_image=inv_image_sourcemasking)
    else:
        source_mask_final = maskBackgroundSources(residual_final, mask_cr=mask_cr, make_plots=make_plots, plot_plots=plot_plots, 
                                                  maxarea=maxarea_sourcemask, image_path=image_path, final=True, 
                                                  r=5, inv_image=inv_image_sourcemasking)
    print("\n7. Creating required variables ...")
    if manual_nri_mask is not None:
        mask_inner_radius, mask_outer_radius = manual_nri_mask

        mask_model, mask_combined, nri, rout = createRequiredVariables(data, model_final, source_mask_final, total_bckgr, geometry, SN=None,
                                                                make_plots=make_plots, plot_plots=plot_plots, image_path=image_path, 
                                                                mask_inner_radius=mask_inner_radius, mask_outer_radius=mask_outer_radius) 
    elif SN_psfitting is not None:
        mask_model, mask_combined, nri, rout = createRequiredVariables(data, model_final, source_mask_final, total_bckgr, geometry, SN=SN_psfitting, globalrms=rms_bckgr,
                                                                make_plots=make_plots, plot_plots=plot_plots, image_path=image_path)        
    else:
        print('Choose better nri mask selection')
        mask_model, mask_combined, nri, rout = createRequiredVariables(data, model_final, source_mask_final, total_bckgr, geometry, SN=None,
                                                                make_plots=make_plots, plot_plots=plot_plots, image_path=image_path)  
    Npix = mask_combined.sum()
    if Npix is None:
        Npix = 0
    if Npix == 0:
        return None, None, None, None, None,None,None, None, None, None, None, None, Npix
    print("\n8. Calculate power spectra  ...")
    image_ps, expected_ps, sbf, noise, std_p0, kfit_i, kfit_f = calculateSBF(nri, mask_combined, psf,
                                                    norm_type = "MaskedPixels",
                                                    fit_range_i=[0.1,0.15,0.2], fit_range_f=[0.6,0.7,0.8],  
                                                    make_plots=make_plots,plot_plots=plot_plots,
                                                    plotyrange=plotyrange,
                                                    image_path=image_path)
    plt.close('all')

    sbfmag = appSBFmagnitude(sbf, mzp)

    if file_path is not None:
        np.savetxt(file_path + "/combined_final_mask", mask_combined)
        np.savetxt(file_path + "/residual_final", residual_final)
        np.savetxt(file_path + "/color_final_mask", mask_model)
        np.savetxt(file_path + "/expected_ps", expected_ps)
        np.savetxt(file_path + "/final_model", model_final)
        geometry_params = np.array((geometry.x0, geometry.y0, geometry.sma, geometry.eps, geometry.pa))
        np.savetxt(file_path + "/geometry", geometry_params)
        np.savetxt(file_path + "/nri", nri)
        np.savetxt(file_path + "/kfit", [kfit_i,kfit_f])
        np.savetxt(file_path + "/sbf", [sbf,std_p0,sbfmag,noise,rout])
        np.savetxt(file_path + "/npix", [Npix])
        
    if fits_path is not None:
        residual_final_masked = np.where(mask_combined.astype(bool), residual_final, 0)
        if os.path.isdir(data_path):
            for file in os.listdir(data_path):
                if ".fits" in file:
                    header = fits.getheader(data_path+ "/" + file)
        elif os.path.isfile(data_path):
            if ".fits" in data_path:
                header = fits.getheader(data_path)
        
        hdu = fits.PrimaryHDU(residual_final, header=header)
                # hdu = fits.open(data_path+ "/" + file)
                # hdu[0].data = residual_final
        hdu.writeto(fits_path+'/residual_final.fits', overwrite=True)
        hdu.data = nri
        hdu.writeto(fits_path+'/nri.fits', overwrite=True)
        hdu.data = residual_final_masked
        hdu.writeto(fits_path+'/residual_final_masked.fits', overwrite=True)
        if galmolcorr == True:
            hdu.data = residual_pre_corr
            hdu.writeto(fits_path+'/residual_pre_corr.fits', overwrite=True)
        del hdu

    if (residual_sources_estimation==True) and (rout>17) and ('field_data' in locals()):     
        print("\n9. Residual sources estimation .. ")

        h = rout+1
        s = 2*h

        patch_model = model_final[geometry.y0-h:geometry.y0+h,geometry.x0-h:geometry.x0+h]
        patch_galaxy_mask = mask_combined[geometry.y0-h:geometry.y0+h,geometry.x0-h:geometry.x0+h]

        field_data -= total_bckgr
        source_mask_field = maskBackgroundSources(field_data, make_plots=False, 
                                                plot_plots=False, maxarea=None, r=7)

        sbf_patches = []
        sbferr_patches = []
        for i in range(4):
            i_str = str(i)
            if i==0:
                patch = field_data[:s,:s]
                patch_mask = ~source_mask_field[:s,:s]
            elif i==1:
                patch = field_data[:s,-s:]
                patch_mask = ~source_mask_field[:s,-s:]
            elif i==2:
                patch = field_data[-s:,:s]
                patch_mask = ~source_mask_field[-s:,:s]
            elif i==3:
                patch = field_data[-s:,-s:]
                patch_mask = ~source_mask_field[-s:,-s:]
                
            #combine with mask from galaxy for determining the relevant sbf region
            patch_mask *= patch_galaxy_mask
            
            patch_norm = patch/np.sqrt(patch_model)
            patch_norm *= patch_mask
            patch_mask *= np.isfinite(patch_norm)
            patch_norm = np.where(np.isfinite(patch_norm), patch_norm,0)
            
            use = True
            use_str = ''
            if patch_mask.sum() < mask_combined.sum()/2:
                if patch_mask.sum() < (50**2)/2:
                    use = False
                    use_str = '_not_used'
            if make_plots:
                if patch_mask.sum() != 0:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    imdisplay(np.ma.masked_array(patch_norm, ~patch_mask), ax, percentlow=1, percenthigh=99, scale='asinh')
                    plt.title(f"Background patch normalised {i_str}")
                    if image_path != None:
                        image_title = f"9.1_background_patch_normalised_{i_str}{use_str}.png"
                        plt.savefig(image_path + "/" + image_title)   
                    if plot_plots:
                        plt.show()
                    plt.close()
            
            if use == True:
                image_ps_patch, expected_ps_patch, sbf_patchi, noise_patch, std_p0_patchi,_,_ = calculateSBF(patch_norm, patch_mask, psf,
                                                                norm_type = "MaskedPixels",
                                                                fit_range_i=0.2, fit_range_f=0.8,
                                                                make_plots=make_plots,plot_plots=plot_plots,
                                                                plotyrange=plotyrange,
                                                                image_path=image_path, image_title=f"9.2_background_patch_power_spectrum_fit_{i_str}.png")
                sbf_patches.append(sbf_patchi)
                sbferr_patches.append(std_p0_patchi)
            elif use == False:
                continue

        sbf_patches = np.array(sbf_patches)
        sbferr_patches = np.array(sbferr_patches)
        if file_path is not None:
            np.savetxt(file_path + "/sbf_patches", [sbf_patches, sbferr_patches])
        if len(sbf_patches) > 0:
            weights = 1 / sbferr_patches**2

            sbf_patch = np.sum(weights * sbf_patches) / np.sum(weights)
            sbferr_patch = np.sqrt(1 / np.sum(weights))

            sbf_cor = sbf-sbf_patch
            sbfmag_cor = appSBFmagnitude(sbf_cor,mzp)
        else:
            print("No proper background patches")
            sbf_patch = None
            sbferr_patch = None
            sbf_cor = None
            sbfmag_cor = None


        del data, model_final, residual_final
        del source_mask, source_mask_final
        del mask_model, mask_combined
        del nri, expected_ps, image_ps

        if 'field_data' in locals():
            del field_data
        if 'patch' in locals():
            del patch, patch_norm, patch_mask

        return sbf, std_p0, sbfmag, noise, sbf_cor, sbferr_patch, sbfmag_cor, geometry.sma, rout, total_bckgr, mzp, geometry, Npix
    else:        
        del data, model_final, residual_final
        del source_mask, source_mask_final
        del mask_model, mask_combined
        del nri, expected_ps, image_ps

        if 'field_data' in locals():
            del field_data
        if 'patch' in locals():
            del patch, patch_norm, patch_mask
        return sbf, std_p0, sbfmag, noise, None,None,None, geometry.sma, rout, total_bckgr, mzp, geometry, Npix






