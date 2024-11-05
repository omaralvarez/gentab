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
            + " & "
            + get_latex_str("HR", g[1], averages, rnks, 2)
            + " & "
            + get_latex_str("EIR", g[1], averages, rnks, 2)
            + " \\\\"
        )
        lines.append(line)

    return lines


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

DCR = pd.DataFrame()
NNDR = pd.DataFrame()
HR = pd.DataFrame()
EIR = pd.DataFrame()

for c in configs:
    dataset = c[1](c[0])
    console.print(dataset.class_counts(), dataset.row_count())

    for g in gens:
        generator = g[0](dataset)
        try:
            generator.load_from_disk()

            min_l2_dists, hits_diffs = dataset.compute_distances_hits(thres_percent=0.3)

            DCR.loc[c[2], g[1]] = dataset.mean_distance_closest_record(min_l2_dists)
            NNDR.loc[c[2], g[1]] = dataset.nearest_neighbor_distance_ratio(min_l2_dists)
            HR.loc[c[2], g[1]] = dataset.hitting_rate(hits_diffs)
            EIR.loc[c[2], g[1]] = dataset.epsilon_identifiability_risk(hits_diffs)

        except FileNotFoundError:
            DCR.loc[c[2], g[1]] = 0.0
            NNDR.loc[c[2], g[1]] = 0.0
            HR.loc[c[2], g[1]] = 1.0
            EIR.loc[c[2], g[1]] = 1.0

round = 2

DCR_mean = DCR.mean()
DCR_ranks = DCR.rank(ascending=True, axis=1)
DCR_mean_rank = DCR_ranks.mean().round(round)

NNDR_mean = NNDR.mean()

HR_mean = HR.mean()

EIR_mean = EIR.mean()

console.print(DCR)
console.print(DCR_mean)
console.print(DCR_ranks)
console.print(NNDR)
console.print(HR)
console.print(EIR)

df = pd.concat(
    [
        DCR_mean_rank.rename("DCR"),
        NNDR_mean.rename("NNDR"),
        HR_mean.rename("HR"),
        EIR_mean.rename("EIR"),
    ],
    axis=1,
)
console.print(df)

ranks = df.rank(method="dense", axis=0, ascending=False)
ranks["HR"] = len(ranks) + 1 - ranks["HR"]
ranks["EIR"] = len(ranks) + 1 - ranks["EIR"]

console.print(ranks)

latex = get_latex(df, ranks, gens)

for line in latex:
    console.print(line)
