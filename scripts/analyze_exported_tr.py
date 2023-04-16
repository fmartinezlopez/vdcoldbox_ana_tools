"""
Created on: 01/03/2023 11:55

Author: Shyam Bhuller

Description: Analyses exported TRs.
"""
import argparse
import os

from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import seaborn as sns

from matplotlib.colors import LogNorm
from rich import print

import detchannelmaps

@dataclass
class Region():
    start_time    : int
    end_time      : int
    start_channel : int
    end_channel   : int
    n_TP          : int

    @property
    def time_range(self):
        return self.end_time - self.start_time
    
    @property
    def channel_range(self):
        return self.end_channel - self.start_channel


class Record:
    def __init__(self, info = None, raw_adcs = None, fwtps = None, tps = None):
        self.info = info
        self.raw_adcs = raw_adcs
        self.fwtps = fwtps
        self.tps = tps


    def AddPlaneID(self, channel_map : str):
        cmap = detchannelmaps.make_map(channel_map)
        self.raw_adcs.columns = pd.MultiIndex.from_tuples([(c, cmap.get_plane_from_offline_channel(c)) for c in self.raw_adcs.columns], names = ["offline_channel", "planeID"])
        if self.tps is not None: self.tps = pd.concat([self.tps, pd.DataFrame({"planeID" : [cmap.get_plane_from_offline_channel(c) for c in self.tps.offline_ch.values]})], axis = 1)


    def FindTriggeredTPs(self):
        return self.tps[self.tps.start_time.isin(self.info.trigger_timestamp)]


    def EventDisplay(self, path : str = "", find_regions : bool = False, plot_triggered_tp : bool = False):
        fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize=(6.4*2, 4.8*2), sharex = True)
        self.AddPlaneID(args.channel_map) # so we can filter by plane ID when making plots #? perhaps this should be done at __init__ or something

        if plot_triggered_tp and self.tps is not None:
            triggered_tp = self.FindTriggeredTPs()
        else:
            triggered_tp = pd.DataFrame()

        y_labels = ["U", "V", "Y"]
        xlim = None
        ylim = None
        for i in range(3):
            adc_df = self.raw_adcs.iloc[:, self.raw_adcs.columns.get_level_values("planeID") == i]
            adc_df.columns = adc_df.columns.get_level_values("offline_channel").values
            start_time = adc_df.index[0]

            adc_df = adc_df.set_index(adc_df.index - start_time)
            adc_df = adc_df - adc_df.median(0) # approximate pedestal subtraction
            adc_df = adc_df[adc_df.columns[::-1]] # reverse column order so channels in ascending order in the y axis after transposing
            rms = np.mean(np.sqrt(np.mean(adc_df**2, axis = 0)))
            sns.heatmap(adc_df.T, ax = axes[-(i + 1)], cmap = "RdGy_r", vmin = -5 * rms, vmax = 5 * rms)


            if self.tps is not None:
                tp_df = self.tps[self.tps.planeID == i]

                if not triggered_tp.empty:
                    triggered_tps_in_plane = triggered_tp[triggered_tp.planeID == i]
                    tp_df = tp_df.drop(list(triggered_tps_in_plane.index))

                x, y = ChannelTimeToPlotSpace(tp_df.peak_time, tp_df.offline_ch, adc_df, start_time) # convert channel and time to heat map coordinates
                if len(y) != len(x):
                    print(f"record {self.info.index} tp conversion didn't work properly")
                else:
                    axes[-(i + 1)].scatter(x, y, color = "dodgerblue", s = 1, label = "TP")

                if not triggered_tp.empty:
                    if not triggered_tps_in_plane.empty:
                        x, y = ChannelTimeToPlotSpace(triggered_tps_in_plane.peak_time, triggered_tps_in_plane.offline_ch, adc_df, start_time)
                        if len(y) != len(x):
                            print(f"record {self.info.index} tp conversion didn't work properly")
                        else:
                            axes[-(i + 1)].scatter(x, y, color = "lime", s = 150, marker = "x", label = "triggered TP")
                        xlim = [x.iloc[0] - 100, x.iloc[0] + 100]
                        ylim = [y[0] + 50, y[0] - 50]


                if find_regions:
                    regions = RegionFinder(tp_df)
                    for r in regions:
                        x, y = ChannelTimeToPlotSpace([r.start_time, r.end_time], [r.start_channel, r.end_channel], adc_df, start_time)
                        rect = patches.Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], linewidth = 1, edgecolor = 'r', facecolor = "none")
                        axes[-(i + 1)].add_patch(rect)

            axes[-(i + 1)].set_xlabel("")
            axes[-(i + 1)].set_ylabel(y_labels[i])

        # trigger 1 run 19562
        # ranges = (
        #     [(850, 1050), (150, 0)], # Y
        #     [(850, 1050), (275, 175)], # V
        #     [(850, 1050), (192, 70)], # U
        # )

        # trigger 121 run 20676
        # ranges = (
        #     [(850, 1050), (170, 20)], # Y
        #     [(850, 1050), (100, 0)], # V
        #     [(850, 1050), (100, 0)], # U
        # )

        # # trigger 123 run 20676
        # ranges = (
        #     [(900, 1250), (150, 0)], # Y
        #     [(900, 1250), (110, 30)], # V
        #     [(900, 1250), (150, 50)], # U
        # )

        # for ax, r in zip(axes, ranges):
        #     ax.set_xlim(r[0])
        #     ax.set_ylim(r[1])

        # axes[2].set_xlabel(f"relative timestamp (ticks)")
        labels = axes[2].get_xticklabels()        
        # print(labels)
        for label in labels:
            label._text = str(int(label._text) * 16/1000)
        # print(labels)
        axes[2].set_xticklabels(labels, rotation = 90)
        axes[2].set_xlabel(f"relative time ($\mu$s)")

        fig.suptitle(f"run: {self.info.run_number.values[0]}, start timestamp : {start_time}")

        handles = []
        labels = []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for i, j in zip(h, l):
                if j not in labels:
                    handles.append(i)
                    labels.append(j)

        if triggered_tp.empty:
            align = 0.5
        else:
            align = 0.525
        fig.legend(handles, labels, bbox_to_anchor=(align, 0), ncol = 2)
        fig.tight_layout()

        name = f"event_display_run_{self.info.run_number.values[0]}_trigger_{self.info.index[0]}.png"
        print(f"Saving event display: {name}")
        fig.savefig(path + name, dpi = 300, bbox_inches = "tight")
        plt.clf()
        plt.close('all')


    def __str__(self):
        return f"{self.info=}\n" + f"{self.raw_adcs=}\n" + f"{self.fwtps=}\n" + f"{self.tps=}\n"


def GetHDFHeirarchy(keys):
    depth = [len(k.split("/")) for k in keys]
    if len(set(depth)) != 1:
        raise Exception("HDF5 file is jagged, not sure what to do here...")

    depth = depth[0] - 1

    sorted_keys = []
    for d in range(depth):
        sorted_keys.append(np.unique([k.split("/")[d + 1] for k in keys]))

    records = []

    for i in range(len(sorted_keys[0])):
        records.append([int(sorted_keys[0][i].split("_")[-1]), sorted_keys[0][i]])
    sorted_keys[0] = np.array(records, dtype = object)
    print(sorted_keys)
    return sorted_keys


def UnpackRecords(hdf : pd.HDFStore, n : list):
    keys = GetHDFHeirarchy(hdf.keys()) # I asusme that keys[0] is the trigger record and keys[1] is the dataframes

    if n == -1:
        n = keys[0][:, 0]

    records = {}
    for i in n:
        if i not in keys[0][:, 0]:
            raise Exception(f"record number {i} not in list of records")
        record = keys[0][keys[0][:, 0] == i][0]
        print(f"opening record {record}")
        r = Record()
        is_empty = False
        for df in keys[1]:
            data = hdf.get("/"+record[1]+"/"+df)
            if data.empty:
                is_empty = True
                break
            setattr(r, df, data)
        if is_empty: continue
        records[record[1]] = r
    return records


def CategoriseTriggers(records : dict):
    n_fakeHSI_trigger = 0
    n_real_trigger = 0

    real_triggers = []
    fake_triggers = []
    for k, v in records.items():
        if v.tps is not None:
            if v.FindTriggeredTPs().empty:
                n_fakeHSI_trigger += 1
                fake_triggers.append(k)
            else:
                n_real_trigger += 1
                real_triggers.append(k)
        else:
            n_fakeHSI_trigger += 1
            fake_triggers.append(k)


    print(f"{n_fakeHSI_trigger=}")
    print(f"{n_real_trigger=}")
    print(f"{fake_triggers=}")
    print(f"{real_triggers=}")
    return fake_triggers, real_triggers


def RegionFinder(tps, time_tolerence = 500, channel_tolerence = 10000, counts_threshold = 1, unique_channel_tolerence = 20) -> list[Region]:
    df = tps.sort_values(by = ["peak_time"])

    # time_tolerence = 23 # how many gaps in time do we tolerate before deeming the end of an object
    # channel_tolerence = 0 # how many gaps in channel do we tolerate before deeming the end of and object

    regions = []
    start_channel = None
    max_channel = None
    min_channel = None

    unique_channels = []

    start_time = None
    nTPs = 1

    for i in range(1, len(df)):
        
        if start_channel is None: start_channel = df.iloc[i - 1].offline_ch
        if start_time is None: start_time = df.iloc[i - 1].peak_time

        if max_channel is None or max_channel < df.iloc[i - 1].offline_ch:
            max_channel = df.iloc[i - 1].offline_ch

        if min_channel is None or min_channel > df.iloc[i - 1].offline_ch:
            min_channel = df.iloc[i - 1].offline_ch

        if len(unique_channels) == 0 or df.iloc[i - 1].offline_ch not in unique_channels:
            unique_channels.append(df.iloc[i - 1].offline_ch)

        # t = (df.iloc[i].peak_time - df.iloc[i - 1].peak_time) // 32
        # c = abs(df.iloc[i].offline_ch - df.iloc[i - 1].offline_ch)
        t = (df.iloc[i].peak_time - start_time) // 32
        c = abs(df.iloc[i].offline_ch - min_channel)

        nTPs += 1

        out_of_time = t > time_tolerence
        out_of_space = c > channel_tolerence
        above_threshold = nTPs > counts_threshold

        if (out_of_time or out_of_space) or i == len(df)-1:
            if above_threshold: # only create a region if there enough TPs in the region
                if len(unique_channels) > unique_channel_tolerence: # only create a region if there are enough unique channels (avoid noisy channels)
                    regions.append(Region(start_time, df.iloc[i].peak_time, min_channel, max_channel, nTPs))
            start_channel = None
            max_channel = None
            start_time = None
            unique_channels = []
            nTPs = 1

    print(f"number of TPs: {len(tps)}")
    print(f"number of regions: {len(regions)}")
    
    total = 0
    for r in regions:
        print(r)
        total += r.n_TP
    print(f"{total=}")
    return regions


def ChannelTimeToPlotSpace(time, channel, heatmap_data, start_time):
    FIR_offest = 16 * 32 # FIR shifts waveform by 16 ticks in firmware then multiply by 32 to scale to WIB frame time ticks
    x = len(heatmap_data.index) * (time - start_time - FIR_offest) / max(heatmap_data.index)
    y = list(map(lambda x : list(np.array(np.where(heatmap_data.columns.values == x)).flatten()), channel)) # maps channels to plot accounting for gaps in the channel numbers

    y_flattened = []
    for i in y:
        y_flattened.extend(i)

    return x, y_flattened


def main(args):
    with pd.HDFStore(args.file) as hdf:
        records = UnpackRecords(hdf, args.n_records)
        fake_triggers, real_triggers = CategoriseTriggers(records)
        
        if len(fake_triggers) > 0:
            path_f = f"run_{records[fake_triggers[0]].info.run_number.values[0]}/fake_triggers/"
            os.makedirs(path_f, exist_ok = True)
            for t in fake_triggers:
                records[t].EventDisplay(path_f)

        if len(real_triggers) > 0:
            path_r = f"run_{records[real_triggers[0]].info.run_number.values[0]}/real_triggers/"
            os.makedirs(path_r, exist_ok = True)
            for t in real_triggers:
                records[t].EventDisplay(path_r)
                # RegionFinder(records[t].tps)


def parse_string(string : str):
    n = [] 
    for s in string.replace(" ", "").split(","):
        r = list(map(int, s.split(":")))
        if len(r) > 1:
            r = range(r[0], r[1] + 1)
        n.extend(r)
    return n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run shower merging analysis in batches", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "Exported dataframe to analyse")
    parser.add_argument("-c", "--channel_map", dest = "channel_map", type = str, choices = [
        "VDColdboxChannelMap",
        "HDColdboxChannelMap",
        "ProtoDUNESP1ChannelMap",
        "PD2HDChannelMap",
        "VSTChannelMap"
    ], help = "channel map to use.", required = True)
    parser.add_argument("-n", "--number_of_records", dest = "n_records", type = str, help = "trigger records to open, -1 opens all", required = True)
    args = parser.parse_args()
    args.n_records = int(args.n_records) if args.n_records == "-1" else parse_string(args.n_records)
    print(vars(args))
    main(args)