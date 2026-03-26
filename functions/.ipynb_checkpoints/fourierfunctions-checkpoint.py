##########################################################################
# This document contains all the functions necessary to do the Fourier
# part of the SBF analysis. 
# The main necessary function is the function calculateSBF()
##########################################################################

# Necessary packages
import numpy as np
import scipy.stats as stats
from scipy.signal import convolve
from functools import partial
from scipy.optimize import curve_fit

def fourier2dPowerSpectrum(image):
    """
    Function that provides the Fourier transform of an image,
    fft-shifted (thus the transform as is usual)
    
    The real, 2d fourier amplitudes are returned.
    """
    non_shifted_transform = np.fft.fft2(image)
    
    fourier_transform = np.fft.fftshift(non_shifted_transform)
    
    fourier_power = np.abs(fourier_transform)**2  
    
    return fourier_power


def azimutalFrequencies(npix):
    """
    Function that returns the 2-dimensional azimutal fourier 
    frequencies, fft shifted. Is used to generate the azimutally 
    averaged power spectrum. 
    """    
    freq_1d = np.fft.fftfreq(npix) * npix

    freq_2d = np.array(np.meshgrid(freq_1d, freq_1d))+0.5
    freq_normed = np.sqrt(freq_2d[0]**2 + freq_2d[1]**2)
    
    # we have to shift to account for a shifted 2D powerspectrum
    frequencies = np.fft.fftshift(freq_normed)
    
    return frequencies
    
    
def powerSpectrum2dTo1d(fourier_power_2d):
    """
    Function converts the 2-dimensional power spectrum into the 
    azimutally averaged Power Spectrum
    """
    npix = fourier_power_2d.shape[0]
    
    frequencies = azimutalFrequencies(npix)
    
    bin_corners = np.arange(0, npix//2+1)
    
    power_spectrum, _, _ = stats.binned_statistic(frequencies.flatten(), 
                                               fourier_power_2d.flatten(),
                                               statistic = "mean",
                                               bins = bin_corners)
    return power_spectrum


def imageToPowerSpectrum(image):
    """
    Function azimutally averages the 2-dimensional power spectrum,
    yielding the averaged 1-dimensional power spectrum
    """
    fourier_power_2d = fourier2dPowerSpectrum(image)
    
    power_spectrum = powerSpectrum2dTo1d(fourier_power_2d)
    
    return power_spectrum
    

def sizePsf(psf_small, image):
    """
    Function that sizes the point spread function to the size of image, 
    in order to be able to apply the convolution.
    
    Resizes by returing zeroes around the psf.
    """
    n_row, n_col = image.shape
    resized_psf = np.zeros((n_col, n_row))
   
    n0 = int(resized_psf.shape[0]/2)
    n1 = int(psf_small.shape[0]/2)
    n2 = int(psf_small.shape[0])
    
    resized_psf[n0-n1:n0-n1+n2,n0-n1:n0-n1+n2] = psf_small
    
    return resized_psf


def psfAndMaskPowerSpectrum(mask, psf):
    """
    Function that convolves the power spectrum of the mask and the psf, 
    and then calculates the azimutally averaged 1d power spectrum
    
    First, the psf is resized to match the mask size
    """
    psf_sized = sizePsf(psf, mask)
    
    mask_ps_2d = fourier2dPowerSpectrum(mask)
    # We need to normalise one of the two to the psf per pixel.
    psf_ps_2d = fourier2dPowerSpectrum(psf_sized)
    
    convolution_ps = (convolve(mask_ps_2d, psf_ps_2d, mode="same")/(mask.shape[0]**2))
    
    power_spectrum_1d = powerSpectrum2dTo1d(convolution_ps)
    
    return power_spectrum_1d
    
    
def sbfToFit(k, P0, P1, Ek):
    """
    Function to fit the Surface Brightness Fluctuations to the Fourier
    1d averaged power spectrum.
    
    The naming conventions of Equation 1 of Jensen et al. (2021) are used. 
    """
    Pk = P0 * Ek + P1
    return Pk


def getNormFactor(norm_type, mask):
    """
    Function that returns a normalisation factor based on the norm_type
    
    There are three options
    0. None: The normalisation is 1, i.e. no normalisation of the power spectrum
    1. MaskedPixels: The normalisation is the number of unmasked pixels
    2. TotalPixels: The normalisation is the total number of pixels in the image
    """
    if (norm_type == None) or (norm_type == 0):
        return 1
    elif (norm_type == "MaskedPixels") or (norm_type == 1):
        return np.sum(mask)
    elif (norm_type == "TotalPixels") or (norm_type == 2):
        return mask.shape[0] * mask.shape[1]
    else:
        print("""
        No valid norm_type. norm_type must either be 
        norm_type = None
        norm_type = 'MaskedPixels'
        norm_type = 'TotalPixels'
        Now, no normalisation is used.""")
        return 1

    
def findExpectedPS(psf_frames, mask, norm_factor):
    """
    Find the expected power spectrum, given either one psf frame or a list
    of psf frames, of which the mean power spectrum is taken.
    """
    if len(psf_frames) == 0:
        expected_psf = None
    elif len(psf_frames.shape) == 2:
        expected_psf = psfAndMaskPowerSpectrum(mask, psf_frames)/norm_factor
    elif len(psf_frames.shape) == 3:
        expected_psf_list = []
        for psf in psf_frames:
            expected_psf_list.append(psfAndMaskPowerSpectrum(mask, psf)/norm_factor)
        expected_psf = np.array(expected_psf_list)
    return expected_psf

def findExpectedPSold(psf_frames, mask, norm_factor):
    """
    Find the expected power spectrum, given either one psf frame or a list
    of psf frames, of which the mean power spectrum is taken.
    """
    if len(psf_frames) == 0:
        expected_psf = None
    elif len(psf_frames.shape) == 2:
        expected_psf = psfAndMaskPowerSpectrum(mask, psf_frames)/norm_factor
    elif len(psf_frames.shape) == 3:
        expected_psf_list = []
        for psf in psf_frames:
            expected_psf_list.append(psfAndMaskPowerSpectrum(mask, psf)/norm_factor)
        expected_psf = np.median(expected_psf_list, axis=0)
    return expected_psf
     
    
def getSbfComponents(residual_fluctuations, mask, psf, norm_type):
    """
    Function that returns the components required in the power spectrum
    fit. These are:
        - The total image power spectrum
        - The expected power spectrum
    
    The norm factor is only required for display purposes.
    """    
    norm_factor = getNormFactor(norm_type, mask)
    
    # Calculate the azimutally averaged Power Spectrum of the image
    image_ps = imageToPowerSpectrum(residual_fluctuations)/norm_factor
    
    # Calculate the power spectrum of the mask and psf (E(k))
    expected_ps = findExpectedPS(psf, mask, norm_factor)
    return image_ps, expected_ps


def fitSbfComponents(image_ps, expected_ps, kfit_i, kfit_f, make_plots=False,plot_plots=False, image_path=None,
                     image_title=None, plotyrange=None):
    """
    Function that performs the fit of the SBF components, given the 
    total image power spectrum and the expected power spectrum
    """
    # Re-define the function to fit
    sbfToFitAdjusted = partial(sbfToFit, Ek=expected_ps[kfit_i:kfit_f])
    
    # The fit is performed without looking at the x value, this is captured in
    # the expected_ps data.
    fit_params, _ = curve_fit(f=sbfToFitAdjusted, 
                              xdata=np.linspace(0,1,len(image_ps[kfit_i:kfit_f])), 
                              ydata=image_ps[kfit_i:kfit_f], 
                              bounds=[[0, 0],[np.inf, np.inf]])   
    
    sbf, noise = fit_params
    
    if make_plots==True:
        plotSbfAnalysis(image_ps, expected_ps, sbf, noise, kfit_i, kfit_f, 
                        image_path=image_path, image_title=image_title, plot_plots=plot_plots, plotyrange=plotyrange)
        
    return sbf, noise
    

def calculateSBF(residual_fluctuations, mask, psf,
                 norm_type = None, 
                 fit_range_i=0.2, fit_range_f=0.6, make_plots=False,plot_plots=False, plotyrange=None,
                 image_path=None, image_title=None):
    """
    Function that calculates the Surface Brightness Fluctuation 
    amplitude and noise level of a residual image, assuming that it
    has already been masked. As input, an NRI, mask, and psf (not 
    sized to total image size) are required.
    
    kfit_i and kfit_f indicate the indices to which the sbf fit 
    should be performed.
    
    norm_type indicates whether the resulting ps should be normalised 
    and to what factor (=either None, MaskedPixels, or TotalPixels). 
    (Only for display purposes in the plot, does not affect measured 
    sbf amplitude.)
    """
    # print("DO NOT USE RN!")
    image_ps, expected_ps = getSbfComponents(residual_fluctuations, mask, 
                                             psf, norm_type)
    if np.any(expected_ps == None):
        print("No psf sources found, no SBF amplitude fit")
        return image_ps, [0], 0, 0
    
    fit_range_i = np.array((fit_range_i))
    fit_range_f = np.array((fit_range_f))
    kfit_i = (fit_range_i*len(image_ps)).astype(int)
    kfit_f = (fit_range_f*len(image_ps)).astype(int)

    if (kfit_i.size == 1) and (kfit_f.size == 1):
        if len(psf.shape) == 2:
            sbf, noise = fitSbfComponents(image_ps, expected_ps, kfit_i, kfit_f, 
                                        image_path=image_path, image_title=image_title, 
                                        make_plots=make_plots,plot_plots=plot_plots, plotyrange=plotyrange)
            std_sbf = None
        elif len(psf.shape) == 3:
            sbfs = []
            noises = []
            for i in range(len(expected_ps[:,0])):
                sbfi, noisei = fitSbfComponents(image_ps, expected_ps[i,:], kfit_i, kfit_f, 
                                image_path=None, image_title=image_title, 
                                make_plots=False,plot_plots=False, plotyrange=plotyrange)
                sbfs.append(sbfi)
                noises.append(noisei)
            sbfs = np.array(sbfs)
            noises = np.array(noises)
            med_sbf = np.median(sbfs)
            std_sbf = np.std(sbfs)
            idx = np.argmin(np.abs(sbfs - med_sbf))
            # sbf = sbfs[idx]
            # noise = noises[idx]
            expected_ps = expected_ps[idx,:]
            sbf, noise = fitSbfComponents(image_ps, expected_ps, kfit_i, kfit_f, 
                                            image_path=image_path, image_title=image_title, 
                                            make_plots=make_plots,plot_plots=plot_plots, plotyrange=plotyrange)
        kfit_i_final = kfit_i
        kfit_f_final = kfit_f

    else:
        
        if kfit_i.size == 1:
            kfit_i = [kfit_i]
        if kfit_f.size == 1:
            kfit_f = [kfit_f]
        sbf_list = []
        ifs = []
        for kfit_i_iter in kfit_i:
            for kfit_f_iter in kfit_f:
                if len(psf.shape) == 2:
                    sbf, noise = fitSbfComponents(image_ps, expected_ps, kfit_i, kfit_f, 
                                                image_path=image_path, image_title=image_title, 
                                                make_plots=make_plots,plot_plots=plot_plots, plotyrange=plotyrange)
                    std_sbf = None
                elif len(psf.shape) == 3:
                    sbfs = []
                    noises = []
                    for i in range(len(expected_ps[:,0])):
                        sbfi, noisei = fitSbfComponents(image_ps, expected_ps[i,:], kfit_i, kfit_f, 
                                        image_path=None, image_title=image_title, 
                                        make_plots=False,plot_plots=False, plotyrange=plotyrange)
                        sbfs.append(sbfi)
                    sbfs = np.array(sbfs)
                    med_sbf = np.median(sbfs)
                    std_sbf = np.std(sbfs)
                    idx = np.argmin(np.abs(sbfs - med_sbf))
                    # sbf = sbfs[idx]
                    # noise = noises[idx]
                    expected_ps = expected_ps[idx,:]
                    sbf, noise = fitSbfComponents(image_ps, expected_ps, kfit_i, kfit_f, 
                                                    image_path=False, image_title=image_title, 
                                                    make_plots=False,plot_plots=False, plotyrange=plotyrange)
                sbf_list.append(sbf)
                ifs.append((kfit_i_iter, kfit_f_iter))

        sbf_list = np.array(sbf_list)
        med_sbf = np.median(sbf_list)
        std_sbf_list = np.std(sbf_list)
        idx = np.argmin(np.abs(sbf_list - med_sbf))
        kfit_i_final = ifs[idx][0]
        kfit_f_final = ifs[idx][1]

        if len(psf.shape) == 2:
            sbf, noise = fitSbfComponents(image_ps, expected_ps, kfit_i_final, kfit_f_final, 
                                        image_path=image_path, image_title=image_title, 
                                        make_plots=make_plots,plot_plots=plot_plots, plotyrange=plotyrange)
            std_sbf = None
        elif len(psf.shape) == 3:
            sbfs = []
            noises = []
            for i in range(len(expected_ps[:,0])):
                sbfi, noisei = fitSbfComponents(image_ps, expected_ps[i,:], kfit_i_final, kfit_f_final, 
                                image_path=None, image_title=image_title, 
                                make_plots=False,plot_plots=False, plotyrange=plotyrange)
                sbfs.append(sbfi)
            sbfs = np.array(sbfs)
            med_sbf = np.median(sbfs)
            std_sbf = np.std(sbfs)
            idx = np.argmin(np.abs(sbfs - med_sbf))
            # sbf = sbfs[idx]
            # noise = noises[idx]
            expected_ps = expected_ps[idx,:]
            sbf, noise = fitSbfComponents(image_ps, expected_ps, kfit_i_final, kfit_f_final, 
                                            image_path=image_path, image_title=image_title, 
                                            make_plots=make_plots,plot_plots=plot_plots, plotyrange=plotyrange)


        std_sbf = np.sqrt((std_sbf**2) + (std_sbf_list**2))
    
    return image_ps, expected_ps, sbf, noise, std_sbf, kfit_i_final//len(image_ps), kfit_f_final//len(image_ps)

def appSBFmagnitude(sbf, mzp):
    m = -2.5 * np.log10(sbf) + mzp
    return m

##########################################################################
# For plotting
##########################################################################

import matplotlib.pyplot as plt

def plotSbfAnalysis(image_ps, expected_ps, sbf, noise, kfit_i, kfit_f,
                    image_path=None, image_title=None, plot_plots=True, plotyrange=None):
    """
    Function that makes a plot of the fitted Fourier Power Spectrum SBF analysis
    """
    fig = plt.figure(figsize=[8,8])
    frame = fig.add_subplot(111)
    
    xrange = np.arange(len(image_ps))
    
    frame.plot(xrange, image_ps, '.')

    frame.plot(xrange, sbfToFit(xrange, sbf, noise, expected_ps),
              label="sbf + noise")
    frame.plot(xrange, sbfToFit(xrange, sbf, 0, expected_ps),
              label="sbf")
    frame.axhline(noise, label="noise")
    
    frame.axvline(kfit_i, linestyle=":", c="0.5")
    frame.axvline(kfit_f, linestyle=":", c="0.5")

    frame.set_xlim(0, len(image_ps))
    frame.set_yscale("log")
    frame.set_ylim(plotyrange)
    frame.legend(fontsize=15, loc=1)
    frame.grid()
    frame.tick_params(axis='both', labelsize=15)
    frame.set_xlabel(r"Wavenumber k  [px$^{-1}$]", fontsize=17)
    frame.set_ylabel(r"P(k) [e$^-$ s$^{-1}$ px$^-1$]", fontsize=17)
    
    if image_path != None:
        if image_title==None:
            image_title = "8.1_power_spectrum_fit.png"
        plt.savefig(image_path + "/" + image_title)
    if plot_plots:
        plt.show()
    
    return
    