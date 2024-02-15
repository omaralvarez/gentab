from synthtab.evaluators import KNN, LightGBM, XGBoost, MLP
from synthtab.generators import (
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
from synthtab.data import Config, Dataset
from synthtab.utils import console

import pandas as pd
import numpy as np


def preproc_playnet(dataset):
    dataset.reduce_size({
        "left_attack": 0.97,
        "right_attack": 0.97,
        "right_transition": 0.9,
        "left_transition": 0.9,
        "time_out": 0.8,
        "left_penal": 0.5,
        "right_penal": 0.5,
    })
    dataset.merge_classes({
        "attack": ["left_attack", "right_attack"],
        "transition": ["left_transition", "right_transition"],
        "penalty": ["left_penal", "right_penal"],
    })
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
    (TVAE, "TVAE"),
    (CTGAN, "CTGAN"),
    (GaussianCopula, "Gaussian Copula"),
    (CopulaGAN, "Copula GAN"),
    (CTABGAN, "CTAB-GAN"),
    (CTABGANPlus, "CTAB-GAN+"),
    (AutoDiffusion, "AutoDiffusion"),
    (ForestDiffusion, "ForestDiffusion"),
    (Tabula, "Tabula"),
    (GReaT, "GReaT"),
]


DCR = pd.DataFrame()

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())
    first = True

    for g in gens:
        generator = g[0](dataset)
        try:
            generator.load_from_disk()

            min_l2_dist = dataset.distance_closest_record()
            DCR.loc[c[2], g[1]] = np.mean(min_l2_dist)
        except FileNotFoundError:
            DCR.loc[c[2], g[1]] = float("inf")

console.print(DCR)

DCR_mean = DCR.mean()
min = DCR_mean.min()

round = 2
lines = []
for index, row in DCR_mean.items():
    if min == row:
        line = (
            index + " & " + "\\textbf{{{:.{prec}f}}}".format(row, prec=round) + " \\\\"
        )
    elif row != float("inf"):
        line = index + " & " + "{:.{prec}f}".format(row, prec=round) + " \\\\"
    else:
        line = index + " & -" + " \\\\"

    lines.append(line)

for line in lines:
    console.print(line)
