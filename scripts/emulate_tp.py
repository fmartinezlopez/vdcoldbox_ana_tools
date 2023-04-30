import argparse
import os
import sys

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

from analyze_exported_tr import UnpackRecords

import TPGAlgorithm


def main(args):
    with pd.HDFStore(args.file) as hdf:
        records = UnpackRecords(hdf, args.n_records)

        elapsed_time = 0
        for name, record in records.items():
            start_time = min(record.raw_adcs.index)
            end_time = max(record.raw_adcs.index)
            elapsed_time += (end_time - start_time) * CLOCK_TICK_NS / 1E9

        print(f"elapsed_time (s) : {elapsed_time}")

        tps_all = []
        for name, record in records.items():
            print(name)
            data = np.array(record.raw_adcs.values, dtype = int)

            tps = TPGAlgorithm.run(data, 40)

            channels = record.raw_adcs.columns # replace arbitrary channel number with provided ones
            tps.offline_ch = channels[tps.offline_ch.values]

            start_time = record.raw_adcs.index[0]
            time_diff = record.raw_adcs.index[1] - record.raw_adcs.index[0]
            print(start_time)
            print(time_diff)

            tps.start_time = (tps.start_time.values * time_diff) + start_time
            tps.peak_time = (tps.peak_time.values * time_diff) + start_time

            tps_all.append(tps)
            print(tps)

        tps_all = pd.concat(tps_all, axis = 1)
        counts = np.unique(tps_all.offline_ch, return_counts = True)
        channels = [1635,1637,1638,1640,1641,1643,1644,1651,2014,2066,2068,2069,2072,2075,2077,2089,2091,2092,2096,2099,2106,2108]
        print([counts[1][counts[0] == c] for c in channels])


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
    parser.add_argument("-w", "--wib_frame", dest = "wib_frame", type = str, choices = ["ProtoDUNE", "DUNE"], help = "wib frame type at the time of data taking.", required = True)
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

    if args.wib_frame == "ProtoDUNE":
        CLOCK_TICK_NS = 20 # ns
    if args.wib_frame == "DUNE":
        CLOCK_TICK_NS = 16 # ns    

    print(vars(args))
    main(args)