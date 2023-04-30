"""
Created on: 01/03/2023 11:55

Author: Shyam Bhuller

Description: Analyses exported TRs.
"""
import argparse
import configparser
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

import TPGAlgorithm

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
        self.sim_tps = None


    def SimulateTPs(self, hit_threshold : int = 20):
        tps = TPGAlgorithm.run(np.array(self.raw_adcs.values, dtype = int), hit_threshold)

        tps.offline_ch = self.raw_adcs.columns[tps.offline_ch.values] # replace arbitrary channel numbers with provided ones

        # convert times from arbitrary numbers to the timestamps starting from the first.
        start_time = self.raw_adcs.index[0]
        time_diff = self.raw_adcs.index[1] - self.raw_adcs.index[0]

        tps.start_time = (tps.start_time.values * time_diff) + start_time
        tps.peak_time = (tps.peak_time.values * time_diff) + start_time
        self.sim_tps = tps


    def AddPlaneID(self, channel_map : str):
        cmap = detchannelmaps.make_map(channel_map)
        if self.raw_adcs is not None: self.raw_adcs.columns = pd.MultiIndex.from_tuples([(c, cmap.get_plane_from_offline_channel(c)) for c in self.raw_adcs.columns], names = ["offline_channel", "planeID"])
        if self.tps is not None: self.tps = pd.concat([self.tps, pd.DataFrame({"planeID" : [cmap.get_plane_from_offline_channel(c) for c in self.tps.offline_ch.values]})], axis = 1)
        if self.sim_tps is not None: self.sim_tps = pd.concat([self.sim_tps, pd.DataFrame({"planeID" : [cmap.get_plane_from_offline_channel(c) for c in self.sim_tps.offline_ch.values]})], axis = 1)


    def FindTriggeredTPs(self):
        return self.tps[self.tps.start_time.isin(self.info.trigger_timestamp)]


    def EventDisplay(self, path : str = "", plot_triggered_tp : bool = True, simulate_tps : bool = False, hit_threshold : int = 20, config : dict = None):
        fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize=(6.4*2, 4.8*2), sharex = True)
        
        if simulate_tps and self.sim_tps == None:
            self.SimulateTPs(hit_threshold)

        self.AddPlaneID(args.channel_map) # so we can filter by plane ID when making plots #? perhaps this should be done at __init__ or something

        if plot_triggered_tp and self.tps is not None:
            triggered_tp = self.FindTriggeredTPs()
        else:
            triggered_tp = pd.DataFrame()

        y_labels = ["U", "V", "Y"]
        for i in range(3):
            adc_df = self.raw_adcs.iloc[:, self.raw_adcs.columns.get_level_values("planeID") == i]
            adc_df.columns = adc_df.columns.get_level_values("offline_channel").values
            start_time = adc_df.index[0]

            adc_df = adc_df.set_index(adc_df.index - start_time)
            adc_df = adc_df - adc_df.median(0) # approximate pedestal subtraction
            adc_df = adc_df[adc_df.columns[::-1]] # reverse column order so channels in ascending order in the y axis after transposing

            rms = np.mean(np.sqrt(np.mean(adc_df**2, axis = 0)))
            print(rms)

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
                    axes[-(i + 1)].scatter(x, y, color = "dodgerblue", s = 50, label = "TP")

                if not triggered_tp.empty:
                    if not triggered_tps_in_plane.empty:
                        x, y = ChannelTimeToPlotSpace(triggered_tps_in_plane.peak_time, triggered_tps_in_plane.offline_ch, adc_df, start_time)
                        if len(y) != len(x):
                            print(f"record {self.info.index} tp conversion didn't work properly")
                        else:
                            axes[-(i + 1)].scatter(x, y, color = "lime", s = 150, marker = "x", label = "triggered TP")
            if simulate_tps:
                tp_df = self.sim_tps[self.sim_tps.planeID == i]
                x, y = ChannelTimeToPlotSpace(tp_df.peak_time, tp_df.offline_ch, adc_df, start_time)
                if len(y) != len(x):
                    print(f"record {self.info.index} tp conversion didn't work properly")
                else:
                    axes[-(i + 1)].scatter(x, y, facecolors = "none", edgecolors = "darkviolet", s = 50, label = " simulated TP", marker = "s")

            x_range = axes[-(i + 1)].get_xlim()
            y_range = axes[-(i + 1)].get_ylim()
            print(f"{x_range}")
            print(f"{y_range}")

            axes[-(i + 1)].set_xlabel("")
            axes[-(i + 1)].set_ylabel(y_labels[i])
            if config:
                if "range" in config["Time"]:
                    const = len(adc_df.index) / (max(adc_df.index) * TIME_UNIT[1])
                    print(f"{const=}")
                    print(f"{max(adc_df.index)=}")
                    print(f"{len(adc_df.index)=}")
                    x = [min(config["Time"]["range"]), max(config["Time"]["range"])]
                    x = [j * const for j in x]
                    print(x)
                    for j in x:
                        if not (min(x_range) <= int(j) <= max(x_range)):
                            raise Exception(f"time plot range must be within {[min(x_range) / const, max(x_range) / const]} {TIME_UNIT[0]}.")

                    axes[-(i+1)].set_xlim(x)
                if "range" in config[y_labels[i]]:
                    y = [max(config[y_labels[i]]["range"]), min(config[y_labels[i]]["range"])]
                    y = [max(adc_df.columns) - j for j in y]
                    for j in y:
                        if not (min(y_range) <= int(j) <= max(y_range)):
                            raise Exception(f"channel {y_labels[i]} plot range must be within {[min(adc_df.columns), max(adc_df.columns)]}.")
                    axes[-(i+1)].set_ylim(y[0], y[1])

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

        if TIME_UNIT[0] == "ticks":
            axes[2].set_xlabel(f"relative timestamp (ticks)")
        else:
            labels = axes[2].get_xticklabels()
            for label in labels:
                print(int(label._text))
                label._text = str(int(label._text) * TIME_UNIT[1])
            axes[2].set_xticklabels(labels, rotation = 90)
            time_label = "$\mu$s" if TIME_UNIT[0] == "us" else TIME_UNIT[0]
            axes[2].set_xlabel(f"relative time ({time_label})")

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
    return sorted_keys


def UnpackRecords(hdf : pd.HDFStore, n : list):
    keys = GetHDFHeirarchy(hdf.keys()) # I asusme that keys[0] is the trigger record and keys[1] is the dataframes

    if n == -1:
        n = keys[0][:, 0]

    records = {}
    for i in n:
        if i not in keys[0][:, 0]:
            raise Exception(f"record number {i} not in list of records: \n{keys[0][:, 0]}")
        record = keys[0][keys[0][:, 0] == i][0]
        print(f"opening record {record}")
        r = Record()
        is_empty = []
        for df in keys[1]:
            data = hdf.get("/"+record[1]+"/"+df)
            if data.empty:
                is_empty.append(df)
                continue
            setattr(r, df, data)
        if len(is_empty) > 0:
            print(f"missing data in record: {is_empty}")
        if "raw_adcs" in is_empty and "tps" in is_empty:
            print(f"{record[1]} has no data! skipping.")
            continue
        else:
            pass # we have all the data in the record
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


def ChannelTimeToPlotSpace(time, channel, heatmap_data, start_time):
    FIR_offest = 15 * 25 # FIR shifts waveform by 16 ticks in firmware then multiply by 32 to scale to WIB frame time ticks
    x = len(heatmap_data.index) * (time - start_time - FIR_offest) / max(heatmap_data.index)
    y = list(map(lambda x : list(np.array(np.where(heatmap_data.columns.values == x)).flatten()), channel)) # maps channels to plot accounting for gaps in the channel numbers

    y_flattened = []
    for i in y:
        y_flattened.extend(i)

    return x, y_flattened


def ChannelRMS(record : Record, path : str = ""):
    y_labels = ["U", "V", "Y"]

    sns.set_theme()
    noisy_channels = None
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(6.4*2, 4.8*1), sharex = True)
    for i in range(3):
        adc_df = record.raw_adcs.iloc[:, record.raw_adcs.columns.get_level_values("planeID") == i]
        adc_df = adc_df - adc_df.median(0) # approximate pedestal subtraction
        adc_df.columns = adc_df.columns.get_level_values("offline_channel").values

        rms = np.sqrt(np.mean(adc_df**2, axis = 0))
        mean_rms = np.mean(np.sqrt(np.mean(adc_df**2, axis = 0)))

        sns.lineplot(x = adc_df.columns, y = rms, ax = axes, label = y_labels[i])
        axes.set_xlabel("channel")
        axes.set_ylabel("rms")

        # print(rms.to_string())

        if noisy_channels is None:
            noisy_channels = rms[rms > 10]
        else:
            noisy_channels = noisy_channels.append(rms[rms > 10])
    
    sns.scatterplot(x = noisy_channels.index, y = noisy_channels, color = "red", label = "rms > 10")
    axes.set_ylim(0)

    masks = [record.raw_adcs.columns.get_level_values("planeID") == i for i in range(3)]
    mask = masks[0] | masks[1] | masks[2]

    channels = record.raw_adcs.columns.get_level_values("offline_channel").values[mask]

    axes.set_xlim(min(channels), max(channels))    
    name = f"run_{record.info.run_number.values[0]}_trigger_{record.info.index[0]}"

    fig.savefig(path + f"rms_" + name + ".png", dpi = 300, bbox_inches = "tight")
    noisy_channels.to_latex(path + f"noisy_channels_" + name + ".tex")
    return


def main(args):
    with pd.HDFStore(args.file) as hdf:
        records = UnpackRecords(hdf, args.n_records)
        fake_triggers, real_triggers = CategoriseTriggers(records)
        
        if len(fake_triggers) > 0:
            path_f = f"run_{records[fake_triggers[0]].info.run_number.values[0]}/fake_triggers/"
            os.makedirs(path_f, exist_ok = True)
            for t in fake_triggers:
                records[t].EventDisplay(path_f, simulate_tps = args.simulate_tps, hit_threshold = args.hit_threshold, config = args.plot_config)
                ChannelRMS(records[t], path_f)

        if len(real_triggers) > 0:
            path_r = f"run_{records[real_triggers[0]].info.run_number.values[0]}/real_triggers/"
            os.makedirs(path_r, exist_ok = True)
            for t in real_triggers:
                records[t].EventDisplay(path_r, simulate_tps = args.simulate_tps, hit_threshold = args.hit_threshold, config = args.plot_config)
                ChannelRMS(records[t], path_r)


def parse_string(string : str):
    n = [] 
    for s in string.replace(" ", "").split(","):
        r = list(map(int, s.split(":")))
        if len(r) > 1:
            r = range(r[0], r[1] + 1)
        n.extend(r)
    return n


def str_to_list(x):
    l = x.split("[")[-1].split("]")[0].split(",")
    return [float(v) for v in l]


def ParsePlotConfig(config : str):
    parser = configparser.ConfigParser()
    parser.read(config)
    parser = parser._sections
    for c in parser:
        for s in parser[c]:
            parser[c][s] = str_to_list(parser[c][s])
    print(parser)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run shower merging analysis in batches", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "Exported dataframe to analyse")
    parser.add_argument("-w", "--wib_frame", dest = "wib_frame", type = str, choices = ["ProtoDUNE", "DUNE"], help = "wib frame type at the time of data taking.", required = True)
    parser.add_argument("-c", "--channel_map", dest = "channel_map", type = str, choices = [
        "VDColdboxChannelMap",
        "HDColdboxChannelMap",
        "ProtoDUNESP1ChannelMap",
        "PD2HDChannelMap",
        "VSTChannelMap"
    ], help = "channel map to use.", required = True)
    parser.add_argument("-n", "--number_of_records", dest = "n_records", type = str, help = "trigger records to open, -1 opens all", required = True)
    parser.add_argument("-s", "--simulate-tps", dest = "simulate_tps", action = "store_true", help = "simulate TPG based on ADC data.")
    parser.add_argument("-T", "--hit-threshold", dest = "hit_threshold", type = int, default = 20, help = "simulate TPG based on ADC data.")
    parser.add_argument("-C", "--config", dest = "plot_config", type = str, help = "configuration for event displays.")
    parser.add_argument("-t", "--time-units", dest = "time_units", type = str, choices = ["ticks", "ns", "us", "ms", "s"], default = "us", help = "unit of time used for plots (plot configuration should be adjusted accordingly)")

    args = parser.parse_args()
    args.n_records = int(args.n_records) if args.n_records == "-1" else parse_string(args.n_records)

    if args.wib_frame == "ProtoDUNE":
        CLOCK_TICK_NS = 20 # ns
    if args.wib_frame == "DUNE":
        CLOCK_TICK_NS = 16 # ns

    TIME_UNITS = {
        "ticks" : 1,
        "ns" : CLOCK_TICK_NS,
        "us" : CLOCK_TICK_NS / 1E3,
        "ms" : CLOCK_TICK_NS / 1E6,
        "s" : CLOCK_TICK_NS  / 1E9,
    }
    TIME_UNIT = (args.time_units, TIME_UNITS[args.time_units])
    if args.plot_config:
        args.plot_config = ParsePlotConfig(args.plot_config)

    print(vars(args))
    main(args)