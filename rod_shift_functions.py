import os
import sys
sys.path.insert(0,os.path.abspath('../admx_analysis_tools/datatypes_and_database/'))
sys.path.insert(0,os.path.abspath('../admx_analysis_tools/config_file_handling/main_analysis/'))
sys.path.insert(0,os.path.abspath('../admx_analysis_tools/parameter_extraction/mainline_analysis/'))
sys.path.insert(0,os.path.abspath('../admx_analysis_tools/noise_models/snr_improvement_method/'))
sys.path.insert(0,os.path.abspath('../admx_analysis_tools/config_file_handling/'))
import admx_db_interface
import admx_db_setup
from admx_db_datatypes import PowerSpectrum
from admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5
from config_file_handling import test_timestamp_cuts
from config_file_handling import get_intermediate_data_file_name
from snr_improvement_method import get_snr_improvement_noise_simple,load_hfet_noise,get_snr_improvement_noise_onres
from parameter_functions import searchId, smooth_parameter, convert_magnet_current_to_B_field

from uncertainties import ufloat
import numpy as np
import h5py
import time
import math
from datetime import datetime,timedelta
import yaml
import argparse
import csv
import pytz
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import scipy.signal as sig
from scipy.interpolate import interp1d
from pathlib import Path
import pyfftw

def read_egg(fname,start_freq,norm_func,chop_num,chop_ind):
    '''
    This is Taj's and Leo's modified read_egg. It has a broader BW and takes in a
    generic function for normaliztion. It also segments the time series data and returns
    the data for a particular segment.
    fname: filename
    start_freq: start_freq of rf spectrum
    norm_func: function to be applied to the spectrum to produce units of sigma
    chop_num: number of segments to split the time series data into
    chop_ind: Index of segment to return data from
    '''
    
    try:
        f = h5py.File(fname,'r')
        digitizer_id = int(Path(fname).stem) #
        data = f.get("/streams/stream0/acquisitions/0")
        meta_data = f.get('streams/stream0')
        dims = np.shape(data) #Data entries are recorded as 32-bit floats, i.e. single precision.
        acq_rate = meta_data.attrs['acquisition_rate']
        numrec = dims[0] # numrec = 10000 = 'n_records'
        numbuf = int(dims[1]/2) # Number complex entries = 'record_size'
        numpts = numrec*numbuf
        reshape_data = data[:].reshape(numpts,-1)
        cdata = pyfftw.empty_aligned(numpts, dtype=np.complex64) #initilize array in memory
        cdata = reshape_data[:,0] + 1j*reshape_data[:,1] #complex time series points
        if cdata.size == 20000000: # 100s run1d files are 2e7 points long
            ts = acq_rate/(meta_data.attrs['record_size']*meta_data.attrs['n_records']*meta_data.attrs['sample_size']) #sample time
            Fs = round(1/ts) # sample rate
            # ad-hoc correction for run 1 D
            Fs *= 4
            fft_freq = np.linspace(0,Fs,numpts,endpoint=False)
            freq = fft_freq + start_freq
            chdata = np.reshape(cdata, (-1, int(cdata.size / chop_num)))
            raw_power = np.abs(pyfftw.interfaces.scipy_fftpack.fft(chdata[chop_ind]))**2 # take fft and square for power
            if not norm_func:
                retvals = (fft_freq, freq, raw_power)
            else: 
                norm_power,fit_params = norm_func(fft_freq,raw_power)
                retvals = (fft_freq, freq, norm_power)
                
            return retvals
        else:
            print(f'size {cdata.size} != 2e7')
            return [False], [False], [False]

    except (OSError,ValueError) as err:
        print(err)


def remove_power_excesses(freqs,spectrum,raw_spec,height,n_bins):
    """
    This function accepts a filtered spectrum and removes power excesses above a certain threshold (given by the height parameter). It is used to
    remove power excesses from the background fit, which contribute to overfitting and a reduction in our sensitivity.

    Regions which are removed due to signal excess are replaced with a linear interpolation between the two disjointed sections of the raw spectrum. 
    
    Params:
    freqs = frequencies (I added this)
    spectrum = raw spectrum with sg fit divided out (blue / orange) (dtype = PowerSpectrum)
    raw_spec = actual raw spectrum (blue)
    height = the excess level which triggers removal (dtype = float) (3.5 or 5)
    n_bins = the number of bins removed (dtype = int)
    """

    #need to loop this for removal of more than one excess within a single raw spectrum
    
    #input('Remove power excess triggered. Press enter to trigger behavior.')

    signal = raw_spec

    idx, _ = sig.find_peaks(spectrum, height=height, threshold=None) #axion location
    #idx = idx[0]
    
    #input(f'Peaks fond at {idx}. Press enter to continue.')
    bool_excess = False

    mask_excess = np.zeros(len(signal),dtype = bool) 
    for i in idx:
        idxs = np.arange(i-(n_bins//2),i+(n_bins//2),1)
        #print('before endpoint fixing')
        #print(idxs)
        # remove indices that are outside the scope of the array
        idxs[idxs < 0 ] = 0
        idxs[idxs > len(signal)-1] = len(signal)-1 #final bin in the signal??
        #print('after endpoint fixing')
        #print(idxs)
        #endpts_freqs = np.array([freqs[idxs[0]-1], freqs[idxs[-1]+1]])
        #endpts_signal = np.array([signal[idxs[0]-1], signal[idxs[-1]+1]])
        endpts_freqs = np.array([freqs[idxs[0]], freqs[idxs[-1]]])
        endpts_signal = np.array([signal[idxs[0]], signal[idxs[-1]]])
        #print(endpts_freqs)
        #print(endpts_signal)

        interpolation = interp1d(endpts_freqs, endpts_signal,fill_value='extrapolate')

        interpolated_freqs = np.arange(freqs[idxs[0]], freqs[idxs[-1]], np.median(np.diff(freqs)))
        #print(interpolated_freqs)

        #interpolated_signal = interpolation(freqs[idxs])
        interpolated_signal = interpolation(interpolated_freqs)

        #print(f'len interpolated_signal: {len(interpolated_signal)}')
        new_signal = np.append(signal[:idxs[0]],interpolated_signal)
        #print(f'len new_signal first append: {len(new_signal)}')
        new_signal = np.append(new_signal,signal[idxs[-1]:])
        #print(idxs[-1])
        #print(f'len new_signal second append: {len(new_signal)}')
        
        mask_excess[idxs[0]:idxs[-1]+1] = True

        #print(signal.size)
        signal = new_signal
        #print(signal.size)
        bool_excess = True

    spectrum = signal

    return spectrum,bool_excess,mask_excess


def is_trackable(digid, time, hires_directory, records2, threshold=2.5):

    if time.hour*3600 + time.minute*60 + time.second - time.utcoffset().total_seconds() - 100 >= 24*60*60:
        new_time = time + timedelta(days=1)
        datestr = f'{new_time.year}{str(new_time.month).zfill(2)}{str(new_time.day).zfill(2)}'
    else:
        datestr = f'{time.year}{str(time.month).zfill(2)}{str(time.day).zfill(2)}'

    fname = f'{hires_directory}{datestr}/{digid}.egg'
    
    _, _, raw_spec = read_egg(fname,0,False,50,0)
    
    if raw_spec[0] == False:
        return False
    raw_spec = np.mean(raw_spec.reshape(-1,int(raw_spec.size/100)), axis=1)
    crns = []
    for r2 in records2:
        if time + timedelta(hours=1) < r2[0]:
            break
        if time - timedelta(hours=1) < r2[0]:
            crns.append(r2[1])
    if len(crns) <= 2:
        return False
    total_crn = np.zeros(2000)
    for crn in crns:
        total_crn = total_crn + crn
    total_crn = total_crn / len(crns)
    total_crn = np.mean(total_crn.reshape(-1,int(total_crn.size/100)), axis=1)
    corr_spec = (raw_spec / (50 * 10**10)) / total_crn
    sg_spec = sig.savgol_filter(corr_spec, 31, 4)
    xs = np.arange(0, 100, 1)
    temp = corr_spec / sg_spec
    corr_spec, _, _ = remove_power_excesses(xs, (temp - np.mean(temp)) / np.std(temp), corr_spec, 3.5, 6)
    sg_spec = sig.savgol_filter(corr_spec, 31, 4)
    test_spec = corr_spec - sg_spec
    stdevrat = np.std(sg_spec[20:80]) / np.std(test_spec)
    if stdevrat < threshold:
        return False
    else:
        return True


def get_shifts(digid, time, data_path, records2, threshold = 5):
    '''
    what would need changed on another computer
    '''
    
    time_str = time.isoformat()
    date_int = int(time_str[:4] + time_str[5:7] + time_str[8:10])
    fname = data_path + str(date_int) + "/" + str(digid) + ".egg"
    if(os.path.isfile(fname) == False):
        date_int += 1
        fname = data_path + str(date_int) + "/" + str(digid) + ".egg"
    crns = []
    for r2 in records2:
        if time + timedelta(hours=1) < r2[0]:
            break
        if time - timedelta(hours=1) < r2[0]:
            crns.append(r2[1])
    total_crn = np.zeros(2000)
    for crn in crns:
        total_crn = total_crn + crn
    total_crn = total_crn / len(crns)
    total_crn = np.mean(total_crn.reshape(-1,int(total_crn.size/100)), axis=1)
    segs = np.arange(0, 50, 1)
    subspectra = []
    for s in segs:
        _, _, raw_spec = read_egg(fname,0,False,50,s)
        if raw_spec[0] == False:
            return False, [0]
        raw_spec = np.mean(raw_spec.reshape(-1,int(raw_spec.size/100)), axis=1)
        corr_spec = (raw_spec / (50 * 10**10)) / total_crn
        sg_spec = sig.savgol_filter(corr_spec, 31, 4)
        xs = np.arange(0, 100, 1)
        temp = corr_spec / sg_spec
        corr_spec, _, _ = remove_power_excesses(xs, (temp - np.mean(temp)) / np.std(temp), corr_spec, 3.5, 6)
        sg_spec = sig.savgol_filter(corr_spec, 31, 4)
        subspectra.append(sg_spec - np.mean(sg_spec))
    best_fits = []
    kernel = subspectra[0][25:75][::-1]
    for s in subspectra:
        conv = sig.convolve(kernel, s,mode='valid')
        best_fits.append(np.argmax(conv))
    f0 = best_fits[0]
    shifts = []
    shift_detected = False
    for s in best_fits:
        shifts.append((s-f0) * 2)
        if abs((s-f0) * 2) > threshold:
            shift_detected = True
    return shift_detected, shifts
    
