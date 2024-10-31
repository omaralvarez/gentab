import re
from gentab.utils import console

import pandas as pd
import numpy as np


def get_latex_str(metric, gen, avs, rnks, round):
    avg = avs[metric][gen]
    if rnks[metric][gen] == 1.0:
        return "\\textbf{{{:.{prec}f}}}".format(avg, prec=round)
    elif rnks[metric][gen] == 2.0:
        return "\\underline{{{:.{prec}f}}}".format(avg, prec=round)
    else:
        return "{:.{prec}f}".format(avg, prec=round)


def get_latex(averages, ranks):
    lines = []

    base = averages.iloc[0]

    for a in averages.iterrows():
        if a[1]["MCC"] == base["MCC"]:
            color = "\\rowcolor{lightgray!30!}"
        elif a[1]["MCC"] > base["MCC"]:
            color = "\\rowcolor{SeaGreen3!30!}"
        else:
            color = "\\rowcolor{RosyBrown2!30!}"

        line = (
            color
            + a[0]
            + " & "
            + get_latex_str("Rank", a[0], averages, ranks, 1)
            + " & "
            + get_latex_str("MCC", a[0], averages, ranks, 2)
            + " & "
            + get_latex_str("Acc", a[0], averages, ranks, 1)
            + "\\%  & "
            + get_latex_str("WPr", a[0], averages, ranks, 1)
            + "\\% & "
            + get_latex_str("MPr", a[0], averages, ranks, 1)
            + "\\% & "
            + get_latex_str("WRe", a[0], averages, ranks, 1)
            + "\\% & "
            + get_latex_str("MRe", a[0], averages, ranks, 1)
            + "\\%  & "
            + get_latex_str("WF", a[0], averages, ranks, 1)
            + "\\% & "
            + get_latex_str("MF", a[0], averages, ranks, 1)
            + "\\% \\\\"
        )
        lines.append(line)

    return lines


keys = ["Dataset", "Model"]
metrics = [
    "MCC",
    "Acc",
    "WPr",
    "MPr",
    "WRe",
    "MRe",
    "WF",
    "MF",
]

file = open("mlperf.dat", "r")

df = pd.DataFrame(columns=keys + metrics)
values = []
while True:
    content = file.readline()

    if not content:
        # datasets.append(values)
        break

    if content.find("@") != -1:
        dset = content
    else:
        split = content.split("&")

        row = [
            dset,
            split[0],
            *[
                float(re.findall("\d+\.\d+", num)[0]) if num.find("-") == -1 else 0.0
                for num in split[1:]
            ],
        ]

        # not including 0s in stats
        if not all(v == 0 for v in row[2:]):
            df.loc[len(df)] = row

        # Including 0s in mean
        # df.loc[len(df)] = row


console.print(df)

avgs = df.groupby(["Model"], sort=False).agg({m: "mean" for m in metrics})
rnd_avgs = avgs.round(2)

console.print(avgs)

# Get real ranks to compute final rank
ranks = avgs.rank(method="dense", axis=0, ascending=False)
# Get ranks of rounded to know which rows need bf or underline
# rnd_ranks = rnd_avgs.rank(method="dense", axis=0, ascending=False)

# Real ranks average column
avgs["Rank"] = ranks.mean(axis=1)
rnd_avgs["Rank"] = ranks.mean(axis=1).round(2)

console.print(avgs["Rank"])

# Second round to compute averages including rank
ranks = rnd_avgs.rank(method="dense", axis=0, ascending=False)
ranks["Rank"] = len(ranks) + 1 - ranks["Rank"]
console.print(ranks)

latex = get_latex(avgs, ranks)

for line in latex:
    console.print(line)

file.close()
