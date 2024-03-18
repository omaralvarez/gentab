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
from gentab.evaluators import KNN, LightGBM, XGBoost, MLP
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
    # ("configs/playnet.json", preproc_playnet, "PlayNet"),
    ("configs/adult.json", preproc_adult, "Adult"),
    # ("configs/car_eval_4.json", preproc_car_eval_4, "Car Evaluation"),
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
    (Tabula, "Tabula", TabulaTuner),
    (GReaT, "GReaT", GReaTTuner),
]

evals = [
    (KNN, "KNN"),
    (LightGBM, "LightGBM"),
    (XGBoost, "XGBoost"),
    # (MLP, "MLP"),
]

for c in configs:
    console.print(c[2])
    dataset = c[1](Dataset(Config(c[0])))

    for g in gens:
        for e in evals:
            tune_and_save(dataset, g[0], e[0], g[2])
