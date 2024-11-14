from gentab.evaluators import KNN, LightGBM, XGBoost, MLP
from gentab.generators import (
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


def get_latex_str(metric, gen, avs, rnks, round):
    avg = avs[metric][gen]
    if rnks[metric][gen] == 1.0:
        return "\\textbf{{{:.{prec}f}}}".format(avg, prec=round)
    elif rnks[metric][gen] == 2.0:
        return "\\underline{{{:.{prec}f}}}".format(avg, prec=round)
    else:
        return "{:.{prec}f}".format(avg, prec=round)


def get_latex(averages, rnks, gens, round):
    lines = []
    for g in gens:
        line = (
            g[1]
            + " & "
            + get_latex_str("JSD", g[1], averages, rnks, round)
            + " & "
            + get_latex_str("WS", g[1], averages, rnks, round)
            + " & "
            + get_latex_str("PCC", g[1], averages, rnks, round)
            + " & "
            + get_latex_str("TU", g[1], averages, rnks, round)
            + " & "
            + get_latex_str("CR", g[1], averages, rnks, round)
            + " \\\\"
        )
        lines.append(line)

    return lines


def preproc_playnet(path):
    dataset = Dataset(Config(path))
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


def preproc_adult(path):
    dataset = Dataset(Config(path))
    dataset.merge_classes({"<=50K": ["<=50K."], ">50K": [">50K."]})

    return dataset


def preproc_car(path):
    dataset = Dataset(Config(path))
    return dataset


def preproc_ecoli(path):
    dataset = Dataset(Config(path))
    return dataset


def preproc_sick(path):
    dataset = Dataset(Config(path))
    return dataset


def preproc_california(path):
    labels = ["lowest", "lower", "low", "medium", "high", "higher", "highest"]
    bins = [float("-inf"), 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, float("inf")]

    dataset = Dataset(Config(path), bins=bins, labels=labels)

    return dataset


configs = [
    ("configs/car_evaluation_cr.json", preproc_car, "Car Evaluation"),
    ("configs/playnet_cr.json", preproc_playnet, "PlayNet"),
    ("configs/adult_cr.json", preproc_adult, "Adult"),
    ("configs/ecoli_cr.json", preproc_ecoli, "Ecoli"),
    ("configs/sick_cr.json", preproc_sick, "Sick"),
    ("configs/california_housing_cr.json", preproc_california, "Calif. Housing"),
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
    dataset = c[1](c[0])
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
ws_mean = ws_ranks.mean().round(round)

console.print(js_mean)
console.print(ws_mean)

console.print("Pearson: \n", pearson)
console.print("Theils: \n", theils)
console.print("Ratio: \n", ratio)

ps_mean = pearson.mean().round(round)
tu_mean = theils.mean().round(round)
rt_mean = ratio.mean().round(round)

df = pd.concat(
    [
        js_mean.rename("JSD"),
        ws_mean.rename("WS"),
        ps_mean.rename("PCC"),
        tu_mean.rename("TU"),
        rt_mean.rename("CR"),
    ],
    axis=1,
)

console.print(df)

ranks = df.rank(method="dense", axis=0, ascending=True)
console.print(ranks)

latex = get_latex(df, ranks, gens, round)

for line in latex:
    console.print(line)
