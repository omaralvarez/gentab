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

    return dataset


configs = [
    ("configs/playnet_cr.json", preproc_playnet, "PlayNet"),
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


DCR_mean = pd.DataFrame()
DCR_var = pd.DataFrame()

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())
    first = True

    for g in gens:
        empty_mean = []  # this goes for all datasets and a single model

        generator = g[0](dataset)
        generator.load_from_disk()

        Min_L2_Dist = dataset.distance_closest_record()
        console.print(Min_L2_Dist)
        empty_mean.append(np.mean(Min_L2_Dist))

        DCR_mean[g[1]] = np.mean(empty_mean)
        DCR_var[g[1]] = np.std(empty_mean)

console.print(DCR_mean)
console.print(DCR_var)
