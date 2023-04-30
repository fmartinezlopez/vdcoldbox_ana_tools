"""
Created on: 30/04/2023 12:26

Author: Shyam Bhuller

Description: Simulation of the TPG algorithm in python.
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def pedestal_subtraction(adcs : np.array) -> np.array:
    """ High pass filter, calculates running median of the adc values over time.

    Args:
        adcs (np.array): ADC data, 2D data with dimensions (channel, time).

    Returns:
        np.array: pedestal subtracted ADC data.
    """
    median = adcs[0, :]
    max_accumulator = 10
    accumulator = np.zeros(adcs.shape[1], dtype = int)

    pedsub = []
    for adc in adcs:# loop through adc data in time (apply the filter for every channel at once)
        # define when the accumulator can increase/decrease
        accumulator = np.where(adc > median, accumulator + 1, accumulator)
        accumulator = np.where(adc < median, accumulator - 1, accumulator)

        # define what happens when the accumulator hits the limits
        median = np.where(accumulator == max_accumulator, median + 1, median)
        median = np.where(accumulator == -max_accumulator, median - 1, median)
        accumulator = np.where(accumulator == max_accumulator, 0, accumulator)
        accumulator = np.where(accumulator == -max_accumulator, 0, accumulator)

        pedsub.append(adc - median)
    return np.array(pedsub)


def fir_filter(adcs : np.array) -> np.array:
    """ Low pass filter, uses a finite impulse response (FIR) filter with 32 taps (16 are set to zero).

    Args:
        adcs (np.array): ADC data, 2D data with dimensions (channel, time).

    Returns:
        np.array: FIR filtered ADC data.
    """
    coefficients = [
        0 ,  0,  0, 0, 0,  0,  0,  0,
        2 ,  4,  6, 7, 9, 11, 12, 13,
        13, 12, 11, 9, 7,  6,  4,  2,
        0 ,  0,  0, 0, 0,  0,  0,  0
    ] # coefficients are hardcoded cause I'm lazy, but this should be configurable.
    fir_filtered = []
    for i in range(len(adcs)): # loop through adc data in time (apply the filter for every channel at once)
        tmp = np.zeros(adcs.shape[1], dtype = int) # blank array for new adc data
        for j in range(len(coefficients)): # loop through all FIR coefficients
            index = i - j if i >= j else False # don't apply the FIR filter to the first 32 values of the ADC data in time
            if index:
                tmp += adcs[index] * coefficients[j] # otherwise create the new adc values from the previous values in time (i.e. a running sum)
        fir_filtered.append(tmp >> 6) # apply a bitshift to re-scale the ADC values (needed because the tap values can't be floats)
    return np.array(fir_filtered)


def hit_finding(adcs : np.array, threshold : int = 20) -> pd.DataFrame:
    """ Hit finding algorithm. Defines a hit as ADC values which exceed a fixed threshold.

    Args:
        adcs (np.array): ADC data, 2D data with dimensions (channel, time).
        threshold (int, optional): ADC threshold. Defaults to 20.

    Returns:
        pd.DataFrame: Trigger primitives found in the data.
    """
    tps = {
        "start_time" : [],
        "peak_time" : [],
        "time_over_threshold" : [],
        "sum_adc" : [],
        "peak_adc" : [],
        "offline_ch" : []
    } # data stored for each trigger primitive
    hit = False
    min_tot = 2 # the minimum width in time a hit must have

    tot = 0 # time over the threshold i.e. width of the hit

    st = 0 # start time
    pt = 0 # peak time
    sadc = 0 # sum ADC
    padc = 0 # peak ADC

    #? can this be vectorised in time to speed things up?
    for channel in range(adcs.shape[1]):
        for time in range(adcs.shape[0]):
            if adcs[time, channel] > threshold and hit == False: # if we aren't in a hit and we are above threshold, we are at the start of a hit.
                hit = True
                st = time
            if hit == True: # if we are in a hit, record information.
                tot += 1
                if adcs[time, channel] > padc:
                    padc = adcs[time, channel]
                    pt = time
                sadc += adcs[time, channel]
            if adcs[time, channel] < threshold and hit == True: # if we are in a hit and below threshold, we are at the end of a hit.
                hit = False
                if tot > min_tot:
                    tps["start_time"].append(st)
                    tps["time_over_threshold"].append(tot)
                    tps["peak_time"].append(pt)
                    tps["sum_adc"].append(sadc)
                    tps["peak_adc"].append(padc)
                    tps["offline_ch"].append(channel)
                    tot = 0
                    st = 0
                    pt = 0
                    sadc = 0
                    padc = 0
    return pd.DataFrame(tps) #? should this convert to data frame or be left as a dictionary for performance?


def generate_gaussian(a : float, b : float, c : float, step : int, _min : int = 0, _max : int = 10) -> np.array:
    """ Function to generte a gaussian waveform. For testing purposes.

    Args:
        a (float): amplitude
        b (float): width
        c (float): offset
        step (int): number of adc values to sample
        _min (int, optional): min x value. Defaults to 0.
        _max (int, optional): max x value. Defaults to 10.

    Returns:
        np.array: _description_
    """
    x = np.linspace(_min, _max, step)
    y = a * np.exp(-(x - b)**2) + c
    return np.array(y, dtype = int)


def test(hit_threshold : int = 20, samples : int = 640, channels : int = 2):
    """ Run TPG algorithm test.

    Args:
        hit_threshold (int, optional): ADC threshold for hit finder. Defaults to 20.
        samples (int, optional): number of samples in time. Defaults to 640.
        channels (int, optional): number of channels. Defaults to 2.
    """
    test = np.random.randint(100, 150, (samples, channels))
    signal = generate_gaussian(100, 5, 0, samples // 10)
    signal = np.pad(signal, (0, test.shape[0] - len(signal)))
    test += signal[:, np.newaxis] # generate test data

    # run each part of the TPG indipendantly
    p = pedestal_subtraction(test)
    f = fir_filter(p)
    tps = hit_finding(f, threshold = hit_threshold)

    print(tps)

    # plot waveforms
    fig, axes = plt.subplots(nrows = test.shape[1], ncols = 1, figsize=(6.4*2, 4.8*test.shape[1]), sharex = True)
    if test.shape[1] == 1:
        axes = [axes]
    for i in range(len(axes)):
        axes[i].plot(test[:, i]) # original waveform
        axes[i].plot(p[:, i]) # pedestal subtracted
        axes[i].plot(f[:, i]) # FIR filtered

        axes[i].axhline(hit_threshold, color = "purple", linestyle = ":") # hit threshold

        tp_ch = tps[tps.offline_ch == i]
        [axes[i].axvline(v, color = "black", linestyle = "--") for v in tp_ch.peak_time.values] # plot black lines at the peakADC of each hit

    plt.savefig("test.png")


def run(adcs : np.array, hit_threshold : int) -> pd.DataFrame:
    """ Run TPG.

    Args:
        adcs (np.array): ADC data, 2D data with dimensions (channel, time).
        hit_threshold (int): ADC threshold of hit finder.

    Returns:
        pd.DataFrame : simulated TPs.
    """
    return hit_finding(fir_filter(pedestal_subtraction(adcs)), hit_threshold)