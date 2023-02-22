import numpy as np
import pandas as pd

import operator
import re

import matplotlib.pyplot as plt
import matplotlib.colors

#TODO: fix binning for 1D and 2D histograms

#HARDCODED!!!
#Boundaries of the different CRP planes in offline channel numbers
plane_index = [0, 952, 1904, 3072]

#Dictionary of basic binary operations to parse math expressions with DF columns
operators = {
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    '/' : operator.truediv,  # use operator.div for Python 2
    '%' : operator.mod,
    '^' : operator.xor,
}

def open_adc(filename: str):
    '''
    Reads the given file and extracts the corresponding ADC pandas DataFrame.

            Args:
                    filename  (str):                 Path to HDF5 file with (from dtpfeedbacktools/dtp-feedback-proxy.py).
            
            Returns:
                    adc_df    (pandas.DataFrame):    ADC DataFrame.

    '''

    adc_df = pd.read_hdf(filename, key="raw_adcs")
    return adc_df

def open_fwtps(filename: str):
    '''
    Reads the given file and extracts the corresponding FWTP pandas DataFrame.

            Args:
                    filename  (str):                 TP DataFrame to use.
            
            Returns:
                    fwtp_df     (pandas.DataFrame):    FWTP DataFrame.

    '''

    fwtp_df = pd.read_hdf(filename, key="raw_fwtps")
    return fwtp_df

def open_tps(filename: str):
    '''
    Reads the given file and extracts the corresponding TP pandas DataFrame.

            Args:
                    filename  (str):                 TP DataFrame to use.
            
            Returns:
                    tp_df     (pandas.DataFrame):    TP DataFrame.

    '''

    tp_df = pd.read_hdf(filename, key="tps")
    return tp_df

#This implementation of the rms computation takes a lot of time
#TODO: write a better one!
def rms(values):
    return np.sqrt(sum(values**2)/len(values))

def parse_df_math(df: pd.DataFrame, string: str):
    '''
    Returns the result of a basic binary operation on any two columns of the TP pandas DataFrame.

            Args:
                    df              (pandas.DataFrame):         TP DataFrame to use.
                    string          (str):                      Binary operation between two keys of df without white spaces (e.g. "peak_time-start_time").
            
            Returns:
                    result          (pandas.Series):            Result of the operation on the two columns.

    '''

    split_string = re.split('(\W+)', string)
    result = operators[split_string[1]](df[split_string[0]], df[split_string[2]])
    return result

def plot_hist_stacked(df: pd.DataFrame, var:str, nbins=None, bins=None, xscale="linear", yscale="linear", grid=True, save=False, outfile=None):
    '''
    Produces a 1D histogram using the TP pandas DataFrame provided for any variable.

            Args:
                    df      (pandas.DataFrame):   TP DataFrame to use.
                    var     (str):                Variable to use. Can be a key of tp or a combination of keys.
                    nbins   (int):                Number of bins to use.
                    bins    (list, numpy.array):  Custom binning to use for.
                    xscale  (str):                Scale to use for x-axis (either "linear" or "log").                        ["linear"]
                    yscale  (str):                Scale to use for y-axis (either "linear" or "log").                        ["linear"]
                    grid    (bool):               Use grid in plots background.                                              [True]
                    save    (bool):               Save plot to pdf.                                                          [False]
                    outfile (str):                Name of output file.

    '''

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)

    ax.set_ylabel("Probability")

    dict_style = {0: ("dodgerblue", False), 1: ("red", False), 2: ("gray", True)}

    for i in range(3):
        df_plane = df.loc[(df["offline_ch"] > plane_index[2-i])&(df["offline_ch"] < plane_index[2-i + 1])]

        if var in df.columns:
            df_plane_x = df_plane[var]
        else:
            df_plane_x = parse_df_math(df_plane, var)

        if bins is None and xscale != "log":
            _bins = np.linspace(df_plane_x.min(), df_plane_x.max(), nbins)
        elif bins is None and xscale == "log":
            _bins = np.logspace(np.log10(df_plane_x.min()), np.log10(df_plane_x.max()), nbins)
        else:
            _bins = bins

        ax.hist(df_plane_x, histtype='step', bins=_bins, color=dict_style[2-i][0], fill=dict_style[2-i][1], density=True)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xscale == "log":
        ax.tick_params(axis='x', which='minor', bottom=False)
    if yscale == "log":
        ax.tick_params(axis='y', which='minor', left=False)
    ax.set_xlabel(var)
    ax.grid(grid)

    gray_patch = matplotlib.patches.Patch(color='gray', alpha=0.7, label='Z Plane')
    blue_patch = matplotlib.patches.Patch(color='dodgerblue', alpha=0.8, label='U Plane')
    red_patch = matplotlib.patches.Patch(color='red', alpha=0.7, label='Y Plane')

    legend = ax.legend(fontsize=12, handles=[gray_patch, blue_patch, red_patch])
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_alpha(0.5)
    frame.set_linewidth(0)
    
    if save:
        if outfile is None:
            outfile = f'hist1d_stacked_{var}-{xscale}-{yscale}.pdf'
        plt.savefig(outfile, dpi=500, bbox_inches='tight')

    plt.show()

def plot_hist_by_plane(df, var, nbins=None, bins=None, color="dodgerblue", xscale="linear", yscale="linear", grid=True, save=False, outfile=None):
    '''
    Produces a 1D histogram using the TP pandas DataFrame provided for any variable.

            Args:
                    df      (pandas.DataFrame):   TP DataFrame to use.
                    var     (str):                Variable to use. Can be a key of tp or a combination of keys.
                    nbins   (int):                Number of bins to use.
                    bins    (list, numpy.array):  Custom binning to use for.
                    xscale  (str):                Scale to use for x-axis (either "linear" or "log").                        ["linear"]
                    yscale  (str):                Scale to use for y-axis (either "linear" or "log").                        ["linear"]
                    color   (str):                Color to use.                                                              ["dodgerblue"]
                    grid    (bool):               Use grid in plots background.                                              [True]
                    save    (bool):               Save plot to pdf.                                                          [False]
                    outfile (str):                Name of output file.

    '''

    fig = plt.figure(figsize=(16,4))
    gs = fig.add_gridspec(1, 3, hspace=0.05)
    axs = gs.subplots()

    axs[0].set_ylabel("Count")

    for i in range(3):
        df_plane = df.loc[(df["offline_ch"] > plane_index[i])&(df["offline_ch"] < plane_index[i + 1])]

        if var in df.columns:
            df_plane_x = df_plane[var]
        else:
            df_plane_x = parse_df_math(df_plane, var)

        if bins is None and xscale != "log":
            _bins = np.linspace(df_plane_x.min(), df_plane_x.max(), nbins)
        elif bins is None and xscale == "log":
            _bins = np.logspace(np.log10(df_plane_x.min()), np.log10(df_plane_x.max()), nbins)
        else:
            _bins = bins

        axs[i].hist(df_plane_x, histtype='step', bins=_bins, color=color)
        axs[i].set_xscale(xscale)
        axs[i].set_yscale(yscale)
        if xscale == "log":
            axs[i].tick_params(axis='x', which='minor', bottom=False)
        if yscale == "log":
            axs[i].tick_params(axis='y', which='minor', left=False)
        axs[i].set_xlabel(var)
        axs[i].grid(grid)
    
    if save:
        if outfile is None:
            outfile = f'hist1d_{var}-{xscale}-{yscale}.pdf'
        plt.savefig(outfile, dpi=500, bbox_inches='tight')

    plt.show()

def plot_2dhist_by_plane(df, varx, vary, nxbins=None, nybins=None, xbins=None, ybins=None, xscale="linear", yscale="linear", grid=False, cmap="plasma", save=False, outfile=None):
    '''
    Produces a 2D histogram using the TP pandas DataFrame provided for any two variables.
    If the variable in the y axis is offline channel number, the plot is

            Args:
                    df      (pandas.DataFrame):   TP DataFrame to use.
                    varx    (str):                Variable to use for x-axis. Can be a key of tp or a combination of keys.
                    vary    (str):                Variable to use for y-axis. Can be a key of tp or a combination of keys.
                    nxbins  (int):                Number of bins to use for x-axis.
                    nybins  (int):                Number of bins to use for y-axis.
                    xbins   (list, numpy.array):  Custom binning to use for x-axis.
                    ybins   (list, numpy.array):  Custom binning to use for y-axis.
                    xscale  (str):                Scale to use for x-axis (either "linear" or "log").                        ["linear"]
                    yscale  (str):                Scale to use for y-axis (either "linear" or "log").                        ["linear"]
                    grid    (bool):               Use grid in plots background.                                              [False]
                    cmap    (str):                Colormap to use.                                                           ["plasma"]
                    save    (bool):               Save plot to pdf.                                                          [False]
                    outfile (str):                Name of output file.

    '''

    if vary == "offline_ch":
        fig = plt.figure(figsize=(10,8))
        gs = fig.add_gridspec(3, 1, wspace=0.05)
        axs = gs.subplots()
        axs[-1].set_xlabel(varx)
        for i in range(3):
            axs[i].set_ylabel(vary)
    else:
        fig = plt.figure(figsize=(16,4))
        gs = fig.add_gridspec(1, 3, hspace=0.05)
        axs = gs.subplots()
        for i in range(3):
            axs[i].set_xlabel(varx)
        axs[0].set_ylabel(vary)

    for i in range(3):
        df_plane = df.loc[(df["offline_ch"] > plane_index[i])&(df["offline_ch"] < plane_index[i + 1])]

        if varx in df.columns:
            df_plane_x = df_plane[varx]
        else:
            df_plane_x = parse_df_math(df_plane, varx)

        if vary in df.columns:
            df_plane_y = df_plane[vary]
        else:
            df_plane_y = parse_df_math(df_plane, vary)

        if xbins is None and xscale != "log":
            _xbins = np.linspace(df_plane_x.min(), df_plane_x.max(), nxbins)
        elif xbins is None and xscale == "log":
            _xbins = np.logspace(np.log10(df_plane_x.min()), np.log10(df_plane_x.max()), nxbins)
        else:
            _xbins = xbins
        
        if ybins is None and yscale != "log":
            _ybins = np.linspace(df_plane_y.min(), df_plane_y.max(), nybins)
        elif ybins is None and yscale == "log":
            _ybins = np.logspace(np.log10(df_plane_y.min()), np.log10(df_plane_y.max()), nybins)
        else:
            _ybins = ybins

        if vary == "offline_ch":
            _ybins = np.linspace(plane_index[i], plane_index[i+1], nybins)
            
        axs[i].hist2d(df_plane_x, df_plane_y, bins=[_xbins, _ybins], cmap=cmap, norm=matplotlib.colors.LogNorm(), rasterized=True)
        axs[i].set_xscale(xscale)
        axs[i].set_yscale(yscale)

        if xscale == "log":
            axs[i].tick_params(axis='x', which='minor', bottom=False)
        if yscale == "log":
            axs[i].tick_params(axis='y', which='minor', left=False)

        if grid:
            axs[i].grid(True)
    
    if save:
        if outfile is None:
            outfile = f'hist2d_{varx}-{xscale}_{vary}-{yscale}.pdf'
        plt.savefig(outfile, dpi=500, bbox_inches='tight')

    plt.show()

def plotme_an_ED(df_adc: pd.DataFrame, zeroped=True, save=False, outfile=None):
    """

    Args:
        df_adc   (pd.DataFrame): _description_
        zeroped  (bool):         _description_    [False]
        out_file (str):          _description_
    """

    cmap = matplotlib.cm.RdBu_r

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(2, 1, hspace=0.05)
    axs = gs.subplots(sharex=True)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1) # cmap for raw adc data
    #create common norm
    if zeroped:
        Z = df_adc.to_numpy()[:,1:]
        Z = Z - np.nanmedian(Z, axis = 0)
        rms = np.nanmean(np.sqrt(np.nanmean(Z**2, axis = 0)))
        #update cmap so it's centered at 0
        norm = matplotlib.colors.TwoSlopeNorm(vmin = -5 * rms, vcenter = 0, vmax = 5 * rms)
    
    #first_ts = df_adc.index[ntsamples[0]]
    #last_ts = df_adc.index[ntsamples[1]]

    for i in range(2):

        #Prepare data for plotting 
        df_plane = df_adc.loc[:, plane_index[i]:plane_index[i + 1]]
        #print(df_plane)
        #df_plane = df_plane.loc[first_ts:last_ts]

        # timestamp for the beginning of data capture that t will be plotted relative to
        relative_ts = df_plane.index - df_plane.index[0]

        Z = df_plane.to_numpy()[:,1:]

        #quick cheated pedsub
        if zeroped:
            Z = Z - np.nanmedian(Z, axis = 0)

        #2D plot of the raw ADC data
        im = axs[i].imshow(Z.T, cmap = cmap, aspect = 'auto', origin = 'lower', norm = norm,
                extent = [ min(relative_ts),max(relative_ts), min(df_plane.columns), max(df_plane.columns) ] )
        
        #axs[i].set_ylabel("Offline channel", fontsize=14, labelpad=10)
        axs[i].tick_params(axis='both', which='major', labelsize=14)
        axs[i].grid(False)

    axs[-1].set_xlabel("Relative time [fw tick]", fontsize=14, labelpad=10)
    
    #axs[0].set_title(f"Run number: {run} | start timestamp: {df_adc.index[0]}", fontsize=16, pad=15)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.7])

    cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cbar_ax.set_ylabel(r"ADC", fontsize=16, labelpad=20, rotation=270)
    cbar_ax.tick_params(axis='both', which='major', labelsize=12)

    if save:
        if outfile is None:
            outfile = f'evd.pdf'
        plt.savefig(outfile, bbox_inches='tight', dpi=500)

    plt.show()