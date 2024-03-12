from synthtab.evaluators import KNN, LightGBM, CatBoost, XGBoost, MLP
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
from functools import reduce


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

    return dataset


def preproc_car_eval_4(dataset):
    return dataset


def evaluate_metrics(g, generator, e, df):
    evaluator = e[0](generator)  # TODO njobs -1
    evaluator.evaluate()

    df.loc[g[2]] = (
        evaluator.mcc,
        evaluator.accuracy,
        evaluator.weighted[0],
        evaluator.macro[0],
        evaluator.weighted[1],
        evaluator.macro[1],
        evaluator.weighted[2],
        evaluator.macro[2],
    )

    return evaluator


# TODO Without tuned dataset it does not error out
def get_metrics(dataset, gens, evals, metrics):
    results = []

    for e in evals:
        data = pd.DataFrame(columns=metrics)
        for g in gens:
            generator = g[0](dataset)
            try:
                try:
                    generator.load_from_disk(str(e[2]))
                    evaluator = evaluate_metrics(g, generator, e, data)
                except FileNotFoundError:
                    generator.load_from_disk()
                    evaluator = evaluate_metrics(g, generator, e, data)
            except FileNotFoundError:
                data.loc[g[2]] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        evaluator.evaluate_baseline()
        data.loc["Baseline"] = (
            evaluator.mcc,
            evaluator.accuracy,
            evaluator.weighted[0],
            evaluator.macro[0],
            evaluator.weighted[1],
            evaluator.macro[1],
            evaluator.weighted[2],
            evaluator.macro[2],
        )

        results.append(data)

    console.print(results)

    averages = reduce(lambda x, y: x + y, results)
    averages /= len(evals)
    averages.loc[:, averages.columns != "MCC"] *= 100.0

    maxs = averages.max()
    maxs[maxs.index != "MCC"] = maxs[maxs.index != "MCC"].round(1)
    maxs.loc["MCC"] = maxs.loc["MCC"].round(2)

    averages.loc[:, averages.columns != "MCC"] = averages.loc[
        :, averages.columns != "MCC"
    ].round(1)
    averages["MCC"] = averages["MCC"].round(2)

    return averages, maxs


def get_latex_str(metric, gen, avs, mxs, round):
    avg = avs[metric][gen]
    if avg == mxs[metric]:
        return "\\textbf{{{:.{prec}f}}}".format(avg, prec=round)
    else:
        return "{:.{prec}f}".format(avg, prec=round)


def get_latex(averages, maxs, gens):
    lines = []
    line = (
        "None"
        + " & "
        + get_latex_str("MCC", "Baseline", averages, maxs, 2)
        + " & "
        + get_latex_str("Acc", "Baseline", averages, maxs, 1)
        + "\\%  & "
        + get_latex_str("WPr", "Baseline", averages, maxs, 1)
        + "\\% & "
        + get_latex_str("MPr", "Baseline", averages, maxs, 1)
        + "\\% & "
        + get_latex_str("WRe", "Baseline", averages, maxs, 1)
        + "\\% & "
        + get_latex_str("MRe", "Baseline", averages, maxs, 1)
        + "\\%  & "
        + get_latex_str("WF", "Baseline", averages, maxs, 1)
        + "\\% & "
        + get_latex_str("MF", "Baseline", averages, maxs, 1)
        + "\\% \\\\"
    )
    lines.append(line)
    line = " & "

    for g in gens:
        if averages.loc[g[2]].sum() == 0.0:
            line = g[1] + " & - & - & - & - & - & - & - & - \\\\"
        else:
            line = (
                g[1]
                + " & "
                + get_latex_str("MCC", g[2], averages, maxs, 2)
                + " & "
                + get_latex_str("Acc", g[2], averages, maxs, 1)
                + "\\%  & "
                + get_latex_str("WPr", g[2], averages, maxs, 1)
                + "\\% & "
                + get_latex_str("MPr", g[2], averages, maxs, 1)
                + "\\% & "
                + get_latex_str("WRe", g[2], averages, maxs, 1)
                + "\\% & "
                + get_latex_str("MRe", g[2], averages, maxs, 1)
                + "\\%  & "
                + get_latex_str("WF", g[2], averages, maxs, 1)
                + "\\% & "
                + get_latex_str("MF", g[2], averages, maxs, 1)
                + "\\% \\\\"
            )
        lines.append(line)

    return lines


configs = [
    ("configs/playnet.json", preproc_playnet, "PlayNet"),
    # ("configs/adult.json", preproc_adult, "Adult"),
    # ("configs/car_eval_4.json", preproc_car_eval_4, "Car Evaluation"),
]

gens = [
    (SMOTE, "SMOTE \cite{chawla2002smote}", "SMOTE"),
    (ADASYN, "ADASYN \cite{he2008adasyn}", "ADASYN"),
    (TVAE, "TVAE \cite{xu2019modeling}", "TVAE"),
    (CTGAN, "CTGAN \cite{xu2019modeling}", "CTGAN"),
    (GaussianCopula, "GaussianCopula \cite{patki2016synthetic}", "GaussianCopula"),
    (CopulaGAN, "CopulaGAN \cite{xu2019modeling}", "CopulaGAN"),
    (CTABGAN, "CTAB-GAN \cite{zhao2021ctab}", "CTAB-GAN"),
    (CTABGANPlus, "CTAB-GAN+ \cite{zhao2022ctab}", "CTAB-GAN+"),
    (AutoDiffusion, "AutoDiffusion \cite{suh2023autodiff}", "AutoDiffusion"),
    (
        ForestDiffusion,
        "ForestDiffusion \cite{jolicoeur2023generating}",
        "ForestDiffusion",
    ),
    (Tabula, "Tabula \cite{zhao2023tabula}", "Tabula"),
    (GReaT, "GReaT \cite{borisov2022language}", "GReaT"),
]

evals = [
    (
        LightGBM,
        "\multirow{" + str(len(gens)) + "}{*}{LightGBM \cite{ke2017lightgbm}}",
        "LightGBM",
    ),
    (
        CatBoost,
        "\multirow{"
        + str(len(gens))
        + "}{*}{CatBoost \cite{prokhorenkova2018catboost}}",
        "CatBoost",
    ),
    (
        XGBoost,
        "\multirow{" + str(len(gens)) + "}{*}{XGBoost \cite{chen2016xgboost}}",
        "XGBoost",
    ),
    (
        MLP,
        "\multirow{" + str(len(gens)) + "}{*}{MLP \cite{gorishniy2021revisiting}}",
        "MLP",
    ),
]

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

for c in configs:
    dataset = c[1](Dataset(Config(c[0])))

    averages, maxs = get_metrics(dataset, gens, evals, metrics)
    latex = get_latex(averages, maxs, gens)

    for line in latex:
        console.print(line)
