from gentab.evaluators import KNN, LightGBM, XGBoost, MLP
from gentab.generators import (
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
from gentab.data import Config, Dataset
from gentab.utils import console

import pandas as pd
import numpy as np


def preproc_playnet(dataset):
    dataset.reduce_size(
        {
            "left_attack": 0.97,
            "right_attack": 0.97,
            "right_transition": 0.9,
            "left_transition": 0.9,
            "time_out": 0.8,
            "left_penal": 0.5,
            "right_penal": 0.5,
        }
    )
    dataset.merge_classes(
        {
            "attack": ["left_attack", "right_attack"],
            "transition": ["left_transition", "right_transition"],
            "penalty": ["left_penal", "right_penal"],
        }
    )
    dataset.reduce_mem()

    return dataset


def preproc_adult(dataset):
    dataset.merge_classes({"<=50K": ["<=50K."], ">50K": [">50K."]})
    dataset.reduce_mem()

    return dataset


def preproc_car(dataset):
    return dataset


configs = [
    ("configs/car_evaluation_cr.json", preproc_car, "Car Evaluation"),
    ("configs/playnet_cr.json", preproc_playnet, "PlayNet"),
    ("configs/adult_cr.json", preproc_adult, "Adult"),
]

gens = [
    (TVAE, "TVAE \cite{xu2019modeling}"),
    (CTGAN, "CTGAN \cite{xu2019modeling}"),
    (GaussianCopula, "GaussianCopula \cite{patki2016synthetic}"),
    (CopulaGAN, "CopulaGAN \cite{xu2019modeling}"),
    (CTABGAN, "CTAB-GAN \cite{zhao2021ctab}"),
    (CTABGANPlus, "CTAB-GAN+ \cite{zhao2022ctab}"),
    (AutoDiffusion, "AutoDiffusion \cite{suh2023autodiff}"),
    (ForestDiffusion, "ForestDiffusion \cite{jolicoeur2023generating}"),
    (GReaT, "GReaT \cite{borisov2022language}"),
    (Tabula, "Tabula \cite{zhao2023tabula}"),
]

jensen_shannon = pd.DataFrame()
wasserstein = pd.DataFrame()
pearson = pd.DataFrame()
theils = pd.DataFrame()
ratio = pd.DataFrame()

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())

    for g in gens:
        generator = g[0](dataset)
        try:
            generator.load_from_disk()

            norm_pearson, norm_theils, norm_ratio = dataset.get_correlations()
            pearson.loc[c[2], g[1]] = norm_pearson
            theils.loc[c[2], g[1]] = norm_theils
            ratio.loc[c[2], g[1]] = norm_ratio

            if len(dataset.get_categories()):
                js_dists = dataset.jensen_shannon_distance()
                jensen_shannon.loc[c[2], g[1]] = js_dists.mean()

            if len(dataset.get_continuous()):
                ws_dists = dataset.wasserstein_distance()
                wasserstein.loc[c[2], g[1]] = ws_dists.mean()

        except FileNotFoundError:
            if len(dataset.get_categories()):
                jensen_shannon.loc[c[2], g[1]] = float("inf")
            if len(dataset.get_continuous()):
                wasserstein.loc[c[2], g[1]] = float("inf")

            pearson.loc[c[2], g[1]] = 0.0
            theils.loc[c[2], g[1]] = 0.0
            ratio.loc[c[2], g[1]] = 0.0

round = 2

console.print("Jensen Shannon: \n", jensen_shannon)
console.print("Wasserstein: \n", wasserstein)

js_ranks = jensen_shannon.rank(ascending=True, axis=1)
ws_ranks = wasserstein.rank(ascending=True, axis=1)
console.print("Jensen Shannon: \n", js_ranks)
console.print("Wasserstein \n", ws_ranks)

js_mean = js_ranks.mean().round(round)
js_min = js_mean.min()
ws_mean = ws_ranks.mean().round(round)
ws_min = ws_mean.min()

console.print(js_mean)
console.print(ws_mean)

console.print("Pearson: \n", pearson)
console.print("Theils: \n", theils)
console.print("Ratio: \n", ratio)

ps_mean = pearson.mean().round(round)
ps_min = ps_mean.min()
tu_mean = theils.mean().round(round)
tu_min = tu_mean.min()
rt_mean = ratio.mean().round(round)
rt_min = rt_mean.min()

console.print(ps_mean)
console.print(tu_mean)
console.print(rt_mean)

lines = []
for (index, js), (_, ws), (_, ps), (_, tu), (_, rt) in zip(
    js_mean.items(), ws_mean.items(), ps_mean.items(), tu_mean.items(), rt_mean.items()
):
    if js_min == js:
        line = index + " & " + "\\textbf{{{:.{prec}f}}}".format(js, prec=round) + " & "
    elif js != float("inf"):
        line = index + " & " + "{:.{prec}f}".format(js, prec=round) + " & "
    else:
        line = index + " & - & "

    if ws_min == ws:
        line += "\\textbf{{{:.{prec}f}}}".format(ws, prec=round) + " & "
    elif ws != float("inf"):
        line += "{:.{prec}f}".format(ws, prec=round) + " & "
    else:
        line += " & - & "

    if ps_min == ps:
        line += "\\textbf{{{:.{prec}f}}}".format(ps, prec=round) + " & "
    elif ps != float("inf"):
        line += "{:.{prec}f}".format(ps, prec=round) + " & "
    else:
        line += " & - & "

    if tu_min == tu:
        line += "\\textbf{{{:.{prec}f}}}".format(tu, prec=round) + " & "
    elif tu != float("inf"):
        line += "{:.{prec}f}".format(tu, prec=round) + " & "
    else:
        line += " & - & "

    if rt_min == rt:
        line += "\\textbf{{{:.{prec}f}}}".format(rt, prec=round) + " \\\\"
    elif rt != float("inf"):
        line += "{:.{prec}f}".format(rt, prec=round) + " \\\\"
    else:
        line += "-  \\\\"

    lines.append(line)

for line in lines:
    console.print(line)
