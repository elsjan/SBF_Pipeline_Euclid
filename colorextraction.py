############
# Write intro
###########

import numpy as np


def downscale_mask_majority(mask, factor=3):
    # reshape into blocks of (factor x factor)
    h, w = mask.shape
    new_h = h // factor
    new_w = w // factor
    
    # reshape to: (new_h, factor, new_w, factor)
    blocks = mask.reshape(new_h, factor, new_w, factor)
    
    # count number of True in each block
    true_count = blocks.sum(axis=(1, 3))
    
    # majority: True only if strictly more than half
    # for 3×3 blocks, majority means >=5 Trues
    return true_count >= ((factor * factor) // 2 + 1)

def trim(data, p, do_print=True):
    if p == 0:
        return data
    if do_print:
        print(f"Trim by {p} pixels")
    if p % 2 == 0:
        # Even 
        p = int(p)//2
        data_new = data[p:-p,p:-p]
    else:
        # Odd
        pl = int(p+1)//2
        pr =  int(p-1)//2
        if pr == 0:
            data_new = data[pl:,pl:]
        else:
            data_new = data[pl:-pr,pl:-pr]    
    
    return data_new

def computeSurfaceBrightness(data, mask_combined, mzp=30.132, scale=0.1, gain=86.95309785869, filter=None,EBV=0):
    # expecting background subracted data
    # standard mzp, scale, and gain set to Euclid VIS (LSB)
    # scale: arcsec per pixel
    # gain: e- per ADU
    # unsure on what exactly went into the calculation of mzp --> read the Euclid pipeline paper
    # to use EBV we would also need R (extinction coefficients?), not applied at this moment, since expected to give little change
    if filter != None:
        if filter == "VIS":
            mzp=30.132
            scale=0.1
            gain = 86.95309785869
        elif filter == "H":
            mzp=30.00
            scale=0.3
            gain = 5.451803290107
        else:
            print("unkown filter: ", filter)

    data_us = data*mask_combined
    adu_per_pix = data_us.sum()/len(data_us)
    adu_per_arcsec = adu_per_pix/(scale**2)
    e_per_arcsec = adu_per_arcsec*gain
    sb = mzp - 2.5 * np.log10(e_per_arcsec)
    return sb

def computeColor(dataB, dataR, mask, mzpI, mzpH, do_print=False):
    """Only takes square images, same coordinate center pixel for galaxies and mask is 
    expected to be from the blue (VIS) image data.
    This code is made for Euclid data, where the VIS pixels scale is 0.1 arcsec and NISP
    is 0.3 arcsec, it account for this scaling.""" 
    mask = ~mask.astype(bool)
    if dataB.shape[0] < dataR.shape[0]*3:
        newshape = int(np.floor(dataB.shape[0]/3))
        dataB = trim(dataB, dataB.shape[0]-newshape*3, do_print=do_print)
        dataR = trim(dataR, dataR.shape[0]-newshape, do_print=do_print)
        mask = trim(mask, mask.shape[0]-newshape*3, do_print=do_print)
        if do_print:
            print(f"Shape VIS data: {dataB.shape[0]}")
            print(f"Shape NISP data: {dataR.shape[0]}")
            print(f"Shape mask: {mask.shape[0]}")
    if dataB.shape[0] > dataR.shape[0]*3:
        newshape = int(np.floor(dataR.shape[0]))
        dataB = trim(dataB, dataB.shape[0]-newshape*3, do_print=do_print)
        dataR = trim(dataR, dataR.shape[0]-newshape, do_print=do_print)
        mask = trim(mask, mask.shape[0]-newshape*3, do_print=do_print)
        if do_print:
            print(f"Shape VIS data: {dataB.shape[0]}")
            print(f"Shape NISP data: {dataR.shape[0]}")
            print(f"Shape mask: {mask.shape[0]}")
    mask_smaller = downscale_mask_majority(mask)
    dataBmasked = np.ma.masked_array(dataB, mask)
    dataRmasked = np.ma.masked_array(dataR, mask_smaller)
    countsB = np.nansum(dataBmasked)
    countsR = np.nansum(dataRmasked)
    magB = mzpI-2.5*np.log10(countsB)
    magR = mzpH-2.5*np.log10(countsR)
    if do_print:
        print("magB, magI, amountpixelsB, amountpixelsR, scaledamountpixelsB")
        print(magB, magR, len(dataBmasked), len(dataRmasked), len(dataBmasked)/3)
        print()
    
    return magB-magR,dataBmasked,dataRmasked


def computeColorEqualSize(dataB, dataR, mask, mzpI, mzpH, do_print=False):
    mask = ~mask.astype(bool)
    if dataB.shape[0] != dataR.shape[0]:
        print(f"Image sizes do not match up {dataB.shape[0]} and {dataR.shape[0]}")
        return None
    #     newshape = int(np.floor(dataB.shape[0]/6))
    #     dataB = trim(dataB, dataB.shape[0]-newshape*6)
    #     dataR = trim(dataR, dataR.shape[0]-newshape*2)
    #     mask = trim(mask, mask.shape[0]-newshape*6)
    #     print(dataB.shape[0])
    #     print(dataR.shape[0])
    #     print(mask.shape[0])
    # if dataB.shape[0] > dataR.shape[0]*3:
    #     newshape = int(np.floor(dataR.shape[0]))
    #     dataB = trim(dataB, dataB.shape[0]-newshape*3)
    #     dataR = trim(dataR, dataR.shape[0]-newshape)
    #     mask = trim(mask, mask.shape[0]-newshape*3)
    #     print(dataB.shape[0])
    #     print(dataR.shape[0])
    #     print(mask.shape[0])
    # mask_smaller = downscale_mask_majority(mask)
    dataBmasked = np.ma.masked_array(dataB, mask)
    dataRmasked = np.ma.masked_array(dataR, mask)
    countsB = np.nansum(dataBmasked)
    countsR = np.nansum(dataRmasked)
    magB = mzpI-2.5*np.log10(countsB)
    magR = mzpH-2.5*np.log10(countsR)
    if do_print:
        print("magB, magI, amountpixelsB, amountpixelsR")
        print(magB, magR, len(dataBmasked), len(dataRmasked))
        print()
    return magB-magR

def computeMagnArea(data, mask, filter='VIS', do_print=False):
    if filter == "VIS":
        mzp=30.132
        scale = 0.1
    elif filter == "H":
        mzp=30.00
        scale = 0.3
        mask = downscale_mask_majority(mask)
    else:
        print("unkown filter: ", filter)
    mask = ~mask.astype(bool)
    masked_data = np.ma.masked_array(data, mask)
    counts = np.nansum(masked_data)
    
    area = (len(masked_data.flatten()) - np.isnan(masked_data).sum())*(scale**2)
    print(area)
    mag = mzp -2.5*np.log10(counts)
    mag_arcsec = mzp -2.5*np.log10(counts/area)
    if do_print:
        print("mag")
        print(mag)
    return mag_arcsec, mag
