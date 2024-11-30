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

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1 import ImageGrid


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


configs = [
    ("configs/playnet_cr.json", preproc_playnet, "PlayNet"),
    ("configs/adult_cr.json", preproc_adult, "Adult"),
    ("configs/car_evaluation_cr.json", preproc_car_eval_4, "Car Evaluation"),
    ("configs/ecoli_cr.json", preproc_ecoli, "Ecoli"),
    ("configs/sick_cr.json", preproc_sick, "Sick"),
    ("configs/california_housing_cr.json", preproc_california, "Calif. Housing"),
    ("configs/mushroom_cr.json", preproc_mushroom, "Mushroom"),
    ("configs/oil_cr.json", preproc_oil, "Oil"),
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
    (GReaT, "GReaT"),
    (Tabula, "Tabula"),
]

for c in configs:
    dataset = c[1](c[0])
    console.print(dataset.class_counts(), dataset.row_count())

    corrs = []
    for g in gens:
        generator = g[0](dataset)
        try:
            generator.load_from_disk()
        except FileNotFoundError:
            console.print("🚨 Missing {} generated data.".format(g[1]))
            corrs.append(None)
            continue

        corrs.append(dataset.get_pearson_correlation().fillna(0))

    max_corr = max(map(lambda x: x.values.max() if x is not None else 0, corrs))

    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(2, len(gens) // 2),
        axes_pad=0.8,
        direction="row",
        share_all=True,
        cbar_mode="single",
        cbar_location="right",
    )

    axs, ims = [], []
    i = 0
    for corr in corrs:
        # Iterating over the grid returns the Axes.
        ax = grid.axes_all[i]
        ax.set_title(gens[i][1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        if corr is not None:
            axs.append(ax)
            ims.append(
                ax.imshow(
                    corr,
                    cmap="Greens",
                    interpolation="nearest",
                    vmin=0.0,
                    vmax=max_corr,
                )
            )
        else:
            axs.append(ax)
            l1 = lines.Line2D([0, 1], [0, 1], transform=fig.transFigure, figure=fig)
            l2 = lines.Line2D([0, 1], [1, 0], transform=fig.transFigure, figure=fig)
            ax.plot(
                [0, dataset.num_features()], [0, dataset.num_features()], color="red"
            )  # Diagonal from bottom-left to top-right

            ims.append(
                ax.plot(
                    [0, dataset.num_features()],
                    [dataset.num_features(), 0],
                    color="red",
                )
            )

        i += 1

    cb = fig.colorbar(ims[0], cax=grid.cbar_axes[0], orientation="vertical")
    cb.outline.set_visible(False)

    plt.savefig(
        "figures/Correlation" + c[2] + ".pdf", format="pdf", bbox_inches="tight"
    )
