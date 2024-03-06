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

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid

# TODO RERUN CORRELATION DUE TO BUG IN CONCAT CODE


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

fig = plt.figure(figsize=(20, 10))
# fig.suptitle("PlayNet")

grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(2, len(gens) // 2),
    axes_pad=0.8,
    direction="row",
    share_all=False,
)

axs, ims = [], []

i = 0
for c in configs:
    config = Config(c[0])

    dataset = c[1](Dataset(config))
    console.print(dataset.class_counts(), dataset.row_count())
    first = True

    for g in gens:
        generator = g[0](dataset)
        generator.load_from_disk()
        corr = dataset.get_correlation()

        ax = grid.axes_all[i]

        ax.set_title(g[1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        axs.append(ax)
        # Iterating over the grid returns the Axes.
        ims.append(ax.imshow(corr, cmap="Greens", interpolation="nearest"))
        # plt.heatmap(corr, #row_labels = xlabs, col_labels = ylabs,
        #   ax = ax, cmap = "YlGn", cbarlabel = "Label")

        i += 1

cmap = mpl.cm.Greens
norm = mpl.colors.Normalize(vmin=0, vmax=1)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs,
    orientation="horizontal",
    fraction=0.1,
)

plt.savefig("figures/CorrelationPlaynet.pdf", format="pdf", bbox_inches="tight")
