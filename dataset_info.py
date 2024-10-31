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


def preproc_ecoli(dataset):
    return dataset


def preproc_sick(dataset):
    return dataset


def preproc_sick(dataset):
    return dataset


def preproc_california(dataset):
    return dataset


def preproc_mushroom(dataset):
    dataset.reduce_size({"e": 0.0, "p": 0.6})
    return dataset


l = ["lowest", "lower", "low", "medium", "high", "higher", "highest"]
b = [float("-inf"), 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, float("inf")]

configs = [
    (
        "configs/car_evaluation_cr.json",
        preproc_car,
        "Car Evaluation \cite{misc_car_evaluation_19}",
        None,
        None,
    ),
    (
        "configs/playnet_cr.json",
        preproc_playnet,
        "PlayNet \cite{mures2023comprehensive}",
        None,
        None,
    ),
    ("configs/adult_cr.json", preproc_adult, "Adult \cite{misc_adult_2}", None, None),
    (
        "configs/ecoli.json",
        preproc_ecoli,
        "Ecoli \cite{ding2011diversified}",
        None,
        None,
    ),
    (
        "configs/sick.json",
        preproc_sick,
        "Sick Euthyroid \cite{ding2011diversified}",
        None,
        None,
    ),
    (
        "configs/california_housing.json",
        preproc_california,
        "Calif. Housing \cite{pace1997sparse}",
        l,
        b,
    ),
    (
        "configs/mushroom.json",
        preproc_mushroom,
        "Mushroom \cite{mushroom_73}",
        None,
        None,
    ),
]

info = {}

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config, labels=c[3], bins=c[4]))

    info[c[2]] = dataset.get_info()

console.print(info)

round = 2
lines = []

for name, info in info.items():
    line = name + " & " + "{:.{prec}f} & ".format(info["imbalance"], prec=round)
    line += "{:.{prec}f}K/{:.{prec}f}K & ".format(
        info["train_samples"] / 1000, info["test_samples"] / 1000, prec=round
    )
    line += str(info["continuous"]) + " & " + str(info["categorical"]) + " \\\\"

    lines.append(line)

for line in lines:
    console.print(line)
