from gentab.generators import (
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


def get_latex_str(metric, gen, avs, rnks, round):
    avg = avs[metric][gen]
    if rnks[metric][gen] == 1.0:
        return "\\textbf{{{:.{prec}f}}}".format(avg, prec=round)
    elif rnks[metric][gen] == 2.0:
        return "\\underline{{{:.{prec}f}}}".format(avg, prec=round)
    else:
        return "{:.{prec}f}".format(avg, prec=round)


def get_latex(averages, rnks, gens):
    lines = []
    for g in gens:
        line = (
            g[1]
            + " & "
            + get_latex_str("DCR", g[1], averages, rnks, 2)
            + " & "
            + get_latex_str("NNDR", g[1], averages, rnks, 2)
            + " \\\\"
        )
        lines.append(line)

    return lines


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

DCR = pd.DataFrame()
NNDR = pd.DataFrame()

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())

    for g in gens:
        generator = g[0](dataset)
        try:
            generator.load_from_disk()

            min_l2_dists = dataset.distance_closest_records()

            DCR.loc[c[2], g[1]] = np.mean(min_l2_dists[:, 0])
            NNDR.loc[c[2], g[1]] = np.mean(min_l2_dists[:, 0] / min_l2_dists[:, 1])
        except FileNotFoundError:
            DCR.loc[c[2], g[1]] = 0.0
            NNDR.loc[c[2], g[1]] = 0.0

round = 2

console.print(DCR)

DCR_mean = DCR.mean()
console.print(DCR_mean)

DCR_ranks = DCR.rank(ascending=True, axis=1)
console.print(DCR_ranks)
DCR_mean_rank = DCR_ranks.mean().round(round)
max = DCR_mean_rank.max()

console.print(NNDR)

NNDR_mean = NNDR.mean()

df = pd.concat(
    [DCR_mean_rank.rename("DCR"), NNDR_mean.rename("NNDR")],
    axis=1,
)
console.print(df)

latex = get_latex(df, df.rank(method="dense", axis=0, ascending=False), gens)

for line in latex:
    console.print(line)
