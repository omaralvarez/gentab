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
from gentab.evaluators import LightGBM, XGBoost, CatBoost, MLP, SVM
from gentab.tuners import (
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

from gentab.data import Config, Dataset
from gentab.utils import console


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


def preproc_car_eval_4(path):
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


def preproc_mushroom(path):
    dataset = Dataset(Config(path))
    dataset.reduce_size({"e": 0.0, "p": 0.6})

    return dataset


def preproc_oil(path):
    dataset = Dataset(Config(path))
    return dataset


def tune_and_save(dataset, gen, eval, tun):
    console.print(dataset.class_counts(), dataset.row_count())
    generator = gen(dataset)
    evaluator = eval(generator)
    tuner = tun(evaluator, trials)
    tuner.tune()
    tuner.save_to_disk()
    console.print(dataset.generated_class_counts(), dataset.generated_row_count())


trials = 10

configs = [
    ("configs/car_eval_4.json", preproc_car_eval_4, "Car Evaluation"),
    ("configs/playnet.json", preproc_playnet, "PlayNet"),
    ("configs/adult.json", preproc_adult, "Adult"),
    ("configs/ecoli.json", preproc_ecoli, "Ecoli"),
    ("configs/sick.json", preproc_sick, "Sick"),
    ("configs/california_housing.json", preproc_california, "Calif. Housing"),
    ("configs/mushroom.json", preproc_mushroom, "Mushroom"),
    ("configs/oil.json", preproc_oil, "Oil"),
]

gens = [
    (SMOTE, "SMOTE", SMOTETuner),
    (ADASYN, "ADASYN", ADASYNTuner),
    (TVAE, "TVAE", TVAETuner),
    (CTGAN, "CTGAN", CTGANTuner),
    (GaussianCopula, "GaussianCopula", GaussianCopulaTuner),
    (CopulaGAN, "CopulaGAN", CopulaGANTuner),
    (CTABGAN, "CTAB-GAN", CTABGANTuner),
    (CTABGANPlus, "CTAB-GAN+", CTABGANPlusTuner),
    (AutoDiffusion, "AutoDiffusion", AutoDiffusionTuner),
    (ForestDiffusion, "ForestDiffusion", ForestDiffusionTuner),
    (GReaT, "GReaT", GReaTTuner),
    (Tabula, "Tabula", TabulaTuner),
]

evals = [
    (LightGBM, "LightGBM"),
    (XGBoost, "XGBoost"),
    (CatBoost, "CatBoost"),
    (MLP, "MLP"),
    (SVM, "SVM"),
]

for c in configs:
    console.print(c[2])
    dataset = c[1](c[0])

    for g in gens:
        for e in evals:
            tune_and_save(dataset, g[0], e[0], g[2])
