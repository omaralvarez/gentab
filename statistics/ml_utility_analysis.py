import re
from gentab.utils import console

import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt
from mycolorpy import colorlist as mcp


def func(x, a, b):
    return a * np.log(x) + b


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
        if (
            a[1]["MCC"] == base["MCC"]
            and a[1]["MPr"] == base["MPr"]
            and a[1]["MRe"] == base["MRe"]
            and a[1]["MF"] == base["MF"]
        ):
            color = "\\rowcolor{lightgray!30!}"
        elif (
            a[1]["MCC"] > base["MCC"]
            or a[1]["MPr"] > base["MPr"]
            or a[1]["MRe"] > base["MRe"]
            or a[1]["MF"] > base["MF"]
        ):
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

imb_metrics = [
    "MCC",
    "MPr",
    "MRe",
    "MF",
]

dataset_info = None

with open("dataset_info.json") as f:
    dataset_info = json.load(f)

info_keys = {value["name"]: key for (key, value) in dataset_info.items()}

file = open("ml_utility.dat", "r")

df = pd.DataFrame(columns=keys + metrics)
values = []
while True:
    content = file.readline()

    if not content:
        break

    if content.find("@") != -1:
        print(content)
        dset = content.strip().replace("@", "")
        bsl = None
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
        if bsl is None:
            bsl = row

        if not all(v == 0 for v in row[2:]):
            df.loc[len(df)] = row
        else:
            df.loc[len(df)] = bsl

console.print(df)

maxs = df.groupby(["Dataset"], sort=False)["MCC"].max()
reals = df[df["Model"].str.contains("None")][["Dataset", "MCC"]].set_index("Dataset")
console.print(maxs)
console.print(reals)

diff = maxs - reals["MCC"]
console.print(diff)
console.print(info_keys)

stats = diff.to_frame().apply(
    lambda row: pd.Series(dataset_info[info_keys[row.name]]), axis=1
)
stats["MCC"] = diff
stats["train_samples"] = stats["train_samples"]
stats = stats.sort_values("train_samples")
console.print(stats)

colors = mcp.gen_color(cmap="viridis", n=5)

# set up the figure and Axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121)

_x = stats["train_samples"]
_y = stats["MCC"]

ax1.scatter(_x / 1000.0, _y, s=30, alpha=0.7, edgecolors="k")

# curve fit the test data
fittedParameters, pcov = opt.curve_fit(func, _x, _y, p0=[0.3, 0.01])
x_pred = np.linspace(10, stats["train_samples"].max(), 100)
modelPredictions = func(x_pred, *fittedParameters)

plt.plot(x_pred / 1000.0, modelPredictions, "--", color="k", lw=1.5)
plt.xlabel("Dataset Size (K)")
plt.ylabel("MCC Gain")
plt.savefig("figures/Oversampling.pdf", format="pdf", bbox_inches="tight")

avgs = df.groupby(["Model"], sort=False).agg({m: "mean" for m in metrics})
rnd_avgs = avgs.round(2)

console.print(avgs)

# Get real ranks to compute final rank
ranks = avgs.rank(method="dense", axis=0, ascending=False)
# Get ranks of rounded to know which rows need bf or underline
# rnd_ranks = rnd_avgs.rank(method="dense", axis=0, ascending=False)

# Real ranks average column
avgs["Rank"] = ranks[imb_metrics].mean(axis=1)
rnd_avgs["Rank"] = ranks[imb_metrics].mean(axis=1).round(2)
console.print(avgs["Rank"])

# Second round to compute averages including rank
ranks = rnd_avgs.rank(method="dense", axis=0, ascending=False)
ranks["Rank"] = rnd_avgs["Rank"].rank(method="dense", axis=0, ascending=True)
console.print(ranks)

latex = get_latex(avgs, ranks)

for line in latex:
    console.print(line)

file.close()
