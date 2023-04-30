"""
Created on: 23/04/2023 12:07

Author: Shyam Bhuller

Description: Analyses exported TSs.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rich import print

import analyze_exported_tr

sns.set_theme()
def TPDisplay(record : analyze_exported_tr.Record, path : str = ""):
    y_labels = ["U", "V", "Y"]
    fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize=(6.4*2, 4.8*2), sharex = True)

    for i in range(3):
        tps = record.tps[record.tps.planeID == i]
        relative_time = (tps.peak_time - min(record.tps.start_time)) * CLOCK_TICK_NS /1E6 # ms

        mask = (relative_time > 500) & (relative_time < 550)
        # mask = (relative_time > 300) & (relative_time < 301)
        tps = tps[mask]
        relative_time = relative_time[mask]
        print(tps)

        sc = axes[-(i + 1)].scatter(relative_time * 1000, tps.offline_ch, c = tps.peak_adc, cmap = "magma", s = 1)
        if i == 2: axes[i].set_xlabel("relative time ($\mu s$)")

        axes[-(i + 1)].set_ylabel(y_labels[i])
        fig.colorbar(sc, ax = axes[-(i + 1)], label = "peak ADC")

    name = f"tp_display_run_{record.info.run_number.values[0]}_trigger_{record.info.index[0]}.png"

    fig.savefig(path + name, dpi = 300, bbox_inches = "tight")
    plt.clf()
    plt.close('all')


def TPRate(record : analyze_exported_tr.Record, path : str = ""):
    y_labels = ["U", "V", "Y"]
    max_time = max(record.tps.start_time + record.tps.time_over_threshold)
    min_time = min(record.tps.start_time)
    # elapsed_time = (max_time - min_time) * 16 / 1000
    bins = np.arange(min_time, max_time, 1000 * 1000 / CLOCK_TICK_NS)

    for i in (range(3)):
        tps = record.tps[record.tps.planeID == i]
        x, y = np.histogram(tps.peak_time, bins)
        plt.plot((y[:-1] - y[0]) * CLOCK_TICK_NS / 1000 / 1000, x, label = y_labels[i])
    plt.legend()
    plt.xlabel("time interval (ms)")
    plt.ylabel("Number of tps per ms")
    
    
    name = f"tp_rate_run_{record.info.run_number.values[0]}_trigger_{record.info.index[0]}.png"
    plt.savefig(path + name, dpi = 300, bbox_inches = "tight")
    plt.clf()
    plt.close('all')


def NoisyChannels(record : analyze_exported_tr.Record, path : str = ""):
    y_labels = ["U", "V", "Y"]
    
    time = record.tps.peak_time - min(record.tps.start_time)
    time = time * CLOCK_TICK_NS / 1E6
    # mask = (time > 500) & (time < 550)
    mask = time >= 0
    tps = record.tps[mask]

    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(6.4*2, 4.8), sharex = True)
    for i in range(3):
        n_tps = np.unique(tps[tps.planeID == i]["offline_ch"], return_counts = True)
        plt.plot(n_tps[0], n_tps[1], label = y_labels[i])

    n_tps = np.unique(tps.offline_ch, return_counts = True)
    plt.axhline(np.median(n_tps[1]), linestyle = "--", color = "red", label = "median")
    # plt.axhline(1.1 * np.median(n_tps[1]), linestyle = "--", color = "purple", label = "1.1 x median")
    plt.xlabel("channel")
    plt.ylabel("number of TPs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test.png", dpi = 300, bbox_inches = "tight")

    noisy_channels = [i[n_tps[1] > 30000] for i in n_tps]

    noisy_channels = pd.DataFrame(noisy_channels[1], index = noisy_channels[0])
    print(noisy_channels)

    name = f"run_{record.info.run_number.values[0]}_trigger_{record.info.index[0]}"
    fig.savefig(path + f"tp_count_" + name + ".png", dpi = 300, bbox_inches = "tight")
    noisy_channels.to_latex(path + f"noisy_channels_" + name + ".tex")


def main(args):
    with pd.HDFStore(args.file) as hdf:
        records = analyze_exported_tr.UnpackRecords(hdf, args.n_records)
        for name, record in records.items():
            path = f"run_{record.info.run_number.values[0]}/"
            os.makedirs(path, exist_ok = True)
            
            record.AddPlaneID(args.channel_map)
            print(np.unique(record.tps.planeID.values, return_counts = True))

            max_time = max(record.tps.start_time + record.tps.time_over_threshold)
            min_time = min(record.tps.start_time)
            print(max_time)
            print(min_time)
            print(f" record length (s): {(max_time - min_time) * CLOCK_TICK_NS / 1E9}")
            TPDisplay(record, path)
            TPRate(record, path)
            NoisyChannels(record, path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run shower merging analysis in batches", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest = "file", type = str, help = "Exported dataframe to analyse")
    parser.add_argument("-w", "--wib_frame", dest = "wib_frame", type = str, choices = ["ProtoDUNE", "DUNE"], help = "wib frame type at the time of data taking.")
    parser.add_argument("-c", "--channel_map", dest = "channel_map", type = str, choices = [
        "VDColdboxChannelMap",
        "HDColdboxChannelMap",
        "ProtoDUNESP1ChannelMap",
        "PD2HDChannelMap",
        "VSTChannelMap"
    ], help = "channel map to use.", required = True)
    parser.add_argument("-n", "--number_of_records", dest = "n_records", type = str, help = "trigger records to open, -1 opens all", required = True)
    args = parser.parse_args()
    args.n_records = int(args.n_records) if args.n_records == "-1" else analyze_exported_tr.parse_string(args.n_records)
    print(vars(args))

    if args.wib_frame == "ProtoDUNE":
        CLOCK_TICK_NS = 20 # ns
    if args.wib_frame == "DUNE":
        CLOCK_TICK_NS = 16 # ns    

    main(args)