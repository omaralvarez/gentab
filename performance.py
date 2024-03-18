from tabgen.evaluators import CatBoost, LightGBM, XGBoost, MLP
from tabgen.generators import (
    ROS,
    SMOTE,
    ADASYN,
    TVAE,
    CTGAN,
    GaussianCopula,
    CopulaGAN,
    CTABGAN,
    CTABGANPlus,
    AutoDiffusion,
    ForestDiffusion,
    Tabula,
    GReaT,
)
from tabgen.tuners import (
    SMOTETuner,
    ADASYNTuner,
    TVAETuner,
    CTGANTuner,
    GaussianCopulaTuner,
    CopulaGANTuner,
    CTABGANTuner,
    CTABGANPlusTuner,
    AutoDiffusionTuner,
    ForestDiffusionTuner,
    TabulaTuner,
    GReaTTuner,
)
from tabgen.data import Config, Dataset
from tabgen.utils import console

from pathlib import Path
import os
import json

import pandas as pd
import numpy as np


def timing_to_disk(timing, folder, dataset, generator):
    # Save generator parameters to JSON
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = os.path.join(
        folder,
        str(dataset).lower()
        + "_"
        + str(generator).lower()
        + "_"
        + "baseline"
        + ".json",
    )
    with open(path, "w") as fp:
        json.dump({"train_time": timing[0], "gen_time": timing[1]}, fp, indent=4)


def preproc_adult(dataset):
    dataset.merge_classes({"<=50K": ["<=50K."], ">50K": [">50K."]})
    # dataset.reduce_mem()

    return dataset


configs = [
    ("configs/adult.json", preproc_adult, "Adult"),
]

gens = [
    # (SMOTE, "SMOTE \cite{chawla2002smote}", SMOTETuner),
    # (ADASYN, "ADASYN \cite{he2008adasyn}", ADASYNTuner),
    (TVAE, "TVAE \cite{xu2019modeling}", TVAETuner),
    (CTGAN, "CTGAN \cite{xu2019modeling}", CTGANTuner),
    # (GaussianCopula, "GaussianCopula \cite{patki2016synthetic}", GaussianCopulaTuner),
    (CopulaGAN, "CopulaGAN \cite{xu2019modeling}", CopulaGANTuner),
    (CTABGAN, "CTAB-GAN \cite{zhao2021ctab}", CTABGANTuner),
    (CTABGANPlus, "CTAB-GAN+ \cite{zhao2022ctab}", CTABGANPlusTuner),
    (AutoDiffusion, "AutoDiffusion \cite{suh2023autodiff}", AutoDiffusionTuner),
    (
        ForestDiffusion,
        "ForestDiffusion \cite{jolicoeur2023generating}",
        ForestDiffusionTuner,
    ),
    (GReaT, "GReaT \cite{borisov2022language}", GReaTTuner),
    (Tabula, "Tabula \cite{zhao2023tabula}", TabulaTuner),
]

evals = [(LightGBM, "LightGBM")]

timing_baseline = pd.DataFrame()
timing_tuned = pd.DataFrame()

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())

    for g in gens:
        generator = g[0](dataset)
        for e in evals:
            evaluator = e[0](generator)

            tuner = g[2](evaluator, 0)
            tuned = tuner.get_tuning_info()

            timing_tuned.loc[g[1], "Fit"] = tuned["train_time"]
            timing_tuned.loc[g[1], "Sample"] = tuned["gen_time"]

            baseline = generator.benchmark()
            timing_to_disk(baseline, tuner.folder, dataset, generator)
            timing_baseline.loc[g[1], "Fit"] = baseline[0]
            timing_baseline.loc[g[1], "Sample"] = baseline[1]

round = 2
timing_baseline = timing_baseline.round(2)
timing_tuned = timing_tuned.round(2)

console.print(timing_baseline)
console.print(timing_tuned)

min_bl = timing_baseline.min()
min_tn = timing_tuned.min()

lines = []
for (index, bl), (_, tn) in zip(timing_baseline.iterrows(), timing_tuned.iterrows()):
    if min_bl.loc["Fit"] == bl["Fit"]:
        line = (
            index
            + " & "
            + "\\textbf{{{:.{prec}f}}}".format(bl["Fit"], prec=round)
            + " & "
        )
    elif bl["Fit"] != 0.0:
        line = index + " & " + "{:.{prec}f}".format(bl["Fit"], prec=round) + " & "
    else:
        line = index + " & - & "

    if min_bl.loc["Sample"] == bl["Sample"]:
        line += "\\textbf{{{:.{prec}f}}}".format(bl["Sample"], prec=round) + " & "
    elif bl["Sample"] != 0.0:
        line += "{:.{prec}f}".format(bl["Sample"], prec=round) + " & "
    else:
        line += " & - & "

    if min_tn.loc["Fit"] == tn["Fit"]:
        line += "\\textbf{{{:.{prec}f}}}".format(tn["Fit"], prec=round) + " & "
    elif tn["Fit"] != 0.0:
        line += "{:.{prec}f}".format(tn["Fit"], prec=round) + " & "
    else:
        line += " & - & "

    if min_tn.loc["Sample"] == tn["Sample"]:
        line += "\\textbf{{{:.{prec}f}}}".format(tn["Sample"], prec=round) + " \\\\"
    elif tn["Sample"] != 0.0:
        line += "{:.{prec}f}".format(tn["Sample"], prec=round) + " \\\\"
    else:
        line += " - \\\\"

    lines.append(line)

for line in lines:
    console.print(line)


timing_baseline_hours = (timing_baseline / 60 / 60).round(2)
timing_tuned_hours = (timing_tuned / 60 / 60).round(2)

console.print(timing_baseline_hours)
console.print(timing_tuned_hours)

min_bl_hours = timing_baseline_hours.min()
min_tn_hours = timing_tuned_hours.min()


lines = []
for (index, bl), (_, tn), (_, blh), (_, tnh) in zip(
    timing_baseline.iterrows(),
    timing_tuned.iterrows(),
    timing_baseline_hours.iterrows(),
    timing_tuned_hours.iterrows(),
):
    if min_bl_hours.loc["Fit"] == blh["Fit"]:
        line = (
            index
            + " & "
            + "\\textbf{{{:.{prec}f}}}".format(blh["Fit"], prec=round)
            + " & "
        )
    elif blh["Fit"] != 0.0:
        line = index + " & " + "{:.{prec}f}".format(blh["Fit"], prec=round) + " & "
    else:
        line = index + " & - & "

    if min_bl.loc["Sample"] == bl["Sample"]:
        line += "\\textbf{{{:.{prec}f}}}".format(bl["Sample"], prec=round) + " & "
    elif bl["Sample"] != 0.0:
        line += "{:.{prec}f}".format(bl["Sample"], prec=round) + " & "
    else:
        line += " & - & "

    if min_tn_hours.loc["Fit"] == tnh["Fit"]:
        line += "\\textbf{{{:.{prec}f}}}".format(tnh["Fit"], prec=round) + " & "
    elif tnh["Fit"] != 0.0:
        line += "{:.{prec}f}".format(tnh["Fit"], prec=round) + " & "
    else:
        line += " & - & "

    if min_tn.loc["Sample"] == tn["Sample"]:
        line += "\\textbf{{{:.{prec}f}}}".format(tn["Sample"], prec=round) + " \\\\"
    elif tn["Sample"] != 0.0:
        line += "{:.{prec}f}".format(tn["Sample"], prec=round) + " \\\\"
    else:
        line += " - \\\\"

    lines.append(line)

for line in lines:
    console.print(line)
