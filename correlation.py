from gentab.evaluators import KNN, LightGBM, XGBoost, MLP
from gentab.generators import (
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
from gentab.data import Config, Dataset
from gentab.utils import console

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1 import ImageGrid


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
    (GReaT, "GReaT"),
    (Tabula, "Tabula"),
]

for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())

    corrs = []
    for g in gens:
        generator = g[0](dataset)
        try:
            generator.load_from_disk()
        except FileNotFoundError:
            corrs.append(None)
            continue

        corrs.append(dataset.get_pearson_correlation())

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
            # Iterating over the grid returns the Axes.
            l1 = lines.Line2D([0, 1], [0, 1], transform=fig.transFigure, figure=fig)
            l2 = lines.Line2D([0, 1], [1, 0], transform=fig.transFigure, figure=fig)
            # fig.lines.extend([l1, l2])
            ax.plot(
                [0, dataset.num_features()], [0, dataset.num_features()], color="red"
            )  # Diagonal from bottom-left to top-right

            ims.append(
                ax.plot(
                    [0, dataset.num_features()],
                    [dataset.num_features(), 0],
                    color="red",
                )  # Diagonal from top-left to bottom-right
            )

        i += 1

    cb = fig.colorbar(ims[0], cax=grid.cbar_axes[0], orientation="vertical")
    cb.outline.set_visible(False)

    plt.savefig(
        "figures/Correlation" + c[2] + ".pdf", format="pdf", bbox_inches="tight"
    )
