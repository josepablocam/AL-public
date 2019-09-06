#!/usr/bin/env python3

from argparse import ArgumentParser

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_times(_dir):
    files = glob.glob(os.path.join(_dir, "script*time.txt"))
    times = []
    for f in files:
        script_id = os.path.basename(f).split("_")[1]
        with open(f, "r") as fin:
            time = float(fin.read())
        times.append((script_id, time))
    return pd.DataFrame(times, columns=["script", "time"])


def get_instr_time_stats(no_instr_dir, instr_dir, output_dir):
    no_instr_df = load_times(no_instr_dir)
    no_instr_df = no_instr_df.rename(columns={"time": "time_no_instr"})

    instr_df = load_times(instr_dir)
    instr_df = instr_df.rename(columns={"time": "time_instr"})

    df = pd.merge(no_instr_df, instr_df, how="left", on="script")
    # remove cases that have to be some artifact of our infrastructure
    # i.e. when instrumented version is faster, likely means something failed
    is_ok = df["time_instr"] > df["time_no_instr"]
    print("Removing bad pairs: {}".format((~is_ok).sum()))
    df = df[is_ok]
    df["ratio"] = df["time_instr"] / df["time_no_instr"]
    df_path = os.path.join(output_dir, "time_ratios.csv")
    df.to_csv(df_path, index=False)

    mean_ratio = df.ratio.mean()
    sd_ratio = df.ratio.std()
    print("Num scripts: {}".format(df.shape[0]))
    print("Ratio: {} (+/- {})".format(mean_ratio, sd_ratio))
    print(
        "Mean uninstrumented time (mins): {} (+/- {})".format(
            df.time_no_instr.mean() / 60,
            df.time_no_instr.std() / 60,
        )
    )
    print(
        "Mean instrumented time (mins): {} (+/- {})".format(
            df.time_instr.mean() / 60,
            df.time_instr.std() / 60,
        )
    )

    df = df.sort_values("ratio")
    fracs, buckets = np.histogram(
        df.ratio.values, bins=np.arange(1, 30), normed=True
    )
    fig, ax = plt.subplots(1)
    ax.bar(buckets[:-1], fracs, width=1.0)
    ax.set_xlabel("Ratio of instrumented runtime to original runtime")
    ax.set_ylabel("Fraction of example scripts")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "time_ratios_plot.pdf")
    fig.savefig(plot_path)


def get_args():
    parser = ArgumentParser(description="Plot instrumentation overhead data")
    parser.add_argument(
        "-n",
        "--no_instr",
        type=str,
        help="Directory for no instrumentation times",
    )
    parser.add_argument(
        "-i",
        "--instr",
        type=str,
        help="Directory for instrumentation times",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    get_instr_time_stats(args.no_instr, args.instr, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
