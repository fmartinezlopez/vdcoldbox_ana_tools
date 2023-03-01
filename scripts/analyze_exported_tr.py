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
import seaborn as sns

from matplotlib.colors import LogNorm
from rich import print

import detchannelmaps


class Record:
    def __init__(self, info = None, raw_adcs = None, fwtps = None, tps = None):
        self.info = info
        self.raw_adcs = raw_adcs
        self.fwtps = fwtps
        self.tps = tps


    def AddPlaneID(self, channel_map : str):
        cmap = detchannelmaps.make_map(channel_map)
        self.raw_adcs.columns = pd.MultiIndex.from_tuples([(c, cmap.get_plane_from_offline_channel(c)) for c in self.raw_adcs.columns], names = ["offline_channel", "planeID"])
        self.tps = pd.concat([self.tps, pd.DataFrame({"planeID" : [cmap.get_plane_from_offline_channel(c) for c in self.tps.offline_ch.values]})], axis = 1)


    def FindTriggeredTPs(self):
        return self.tps[self.tps.start_time.isin(self.info.trigger_timestamp)]


    def EventDisplay(self, path : str = ""):
        fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize=(6.4*2, 4.8*2), sharex = True)
        self.AddPlaneID(args.channel_map) # so we can filter by plane ID when making plots #? perhaps this should be done at __init__ or something

        triggered_tp = self.FindTriggeredTPs()

        y_labels = ["U", "V", "X"]
        for i in range(3):
            adc_df = self.raw_adcs.iloc[:, self.raw_adcs.columns.get_level_values("planeID") == i]
            adc_df.columns = adc_df.columns.get_level_values("offline_channel").values
            start_time = adc_df.index[0]

            adc_df = adc_df.set_index(adc_df.index - start_time)
            adc_df = adc_df - adc_df.median(0) # approximate pedestal subtraction
            adc_df = adc_df[adc_df.columns[::-1]] # reverse column order so channels in ascending order in the y axis after transposing
            rms = np.mean(np.sqrt(np.mean(adc_df**2, axis = 0)))
            sns.heatmap(adc_df.T, ax = axes[-(i + 1)], cmap = "RdGy_r", vmin = -5 * rms, vmax = 5 * rms)


            tp_df = self.tps[self.tps.planeID == i]
            x, y = ChannelTimeToPlotSpace(tp_df.peak_time, tp_df.offline_ch, adc_df, start_time) # convert channel and time to heat map coordinates
            axes[-(i + 1)].scatter(x, y, color = "dodgerblue", s = 5, label = "TP")
            
            triggered_tps_in_plane = triggered_tp[triggered_tp.planeID == i]
            x, y = ChannelTimeToPlotSpace(triggered_tps_in_plane.peak_time, triggered_tps_in_plane.offline_ch, adc_df, start_time)
            axes[-(i + 1)].scatter(x, y, color = "lime", s = 40, marker = "x", label = "triggered TP")

            axes[-(i + 1)].set_xlabel("")
            axes[-(i + 1)].set_ylabel(y_labels[i])
        axes[2].set_xlabel(f"relative timestamp")
        fig.suptitle(f"run: {self.info.run_number.values[0]}, start timestamp : {start_time}")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.525, 0), ncol = 2)
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
    print(sorted_keys)
    return sorted_keys


def UnpackRecords(hdf : pd.HDFStore, n : int):
    keys = GetHDFHeirarchy(hdf.keys()) # I asusme that keys[0] is the trigger record andkeys[1] is the dataframes

    if n >= len(keys[0]):
        raise Exception(f"number of records specified to open is greater than the number of records: {len(keys[0])}")

    if n == -1:
        n = len(keys[0])

    records = {}
    for i in range(n):
        record = keys[0][i]
        print(f"opening record {record}")
        r = Record()
        for df in keys[1]:
            setattr(r, df, hdf.get("/"+record+"/"+df))
        records[record] = r
    return records


def CategoriseTriggers(records : dict):
    n_fakeHSI_trigger = 0
    n_real_trigger = 0

    real_triggers = []
    fake_triggers = []
    for k, v in records.items():
        if v.FindTriggeredTPs().empty:
            n_fakeHSI_trigger += 1
            fake_triggers.append(k)
        else:
            n_real_trigger += 1
            real_triggers.append(k)

    print(f"{n_fakeHSI_trigger=}")
    print(f"{n_real_trigger=}")
    print(f"{fake_triggers=}")
    print(f"{real_triggers=}")
    return fake_triggers, real_triggers


def ChannelTimeToPlotSpace(time, channel, heatmap_data, start_time):
    FIR_offest = 16 * 32 # FIR shifts waveform by 16 ticks in firmware then multiply by 32 to scale to WIB frame time ticks
    x = len(heatmap_data.index) * (time - start_time - FIR_offest) / max(heatmap_data.index)
    y = max(heatmap_data.columns) - channel     
    return x, y


def main(args):
    with pd.HDFStore(args.file) as hdf:
        records = UnpackRecords(hdf, args.n_records)
        print(records["record_0"])
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
    parser.add_argument("-n", "--number_of_records", dest = "n_records", type = int, help = "number of trigger records to open", required = True)
    args = parser.parse_args()
    print(vars(args))
    main(args)
