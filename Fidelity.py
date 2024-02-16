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


jensen_shannon = pd.DataFrame()
wasserstein = pd.DataFrame()

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())

    for g in gens:
        generator = g[0](dataset)
        try:
            generator.load_from_disk()

            if len(dataset.get_categories()):
                js_dists = dataset.jensen_shannon_distance()
                jensen_shannon.loc[c[2], g[1]] = js_dists.mean()

            if len(dataset.get_continuous()):
                ws_dists = dataset.wasserstein_distance()
                wasserstein.loc[c[2], g[1]] = ws_dists.mean()

        except FileNotFoundError:
            jensen_shannon.loc[c[2], g[1]] = float("inf")
            wasserstein.loc[c[2], g[1]] = float("inf")

console.print(jensen_shannon)
console.print(wasserstein)

js_mean = jensen_shannon.mean()
js_min = js_mean.min()
ws_mean = wasserstein.mean()
ws_min = ws_mean.min()

console.print(js_mean)
console.print(ws_mean)
