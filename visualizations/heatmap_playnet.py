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

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, Arc
import numpy as np
from scipy.stats import kde


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


gens = [
    (None, "Baseline"),
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

config = Config("configs/playnet_cr.json")

dataset = preproc_playnet(Dataset(config))

gs = gridspec.GridSpec(len(gens), dataset.num_classes())
fig = plt.figure(figsize=(40, 30))
fig.subplots_adjust(wspace=0.25, hspace=0.125)

axs, ims = [], []
maxcnt = 0
i = 0
for g in gens:
    if g[0] is not None:
        generator = g[0](dataset)
        generator.load_from_disk()

    j = 0
    for c in dataset.class_names():
        ax = fig.add_subplot(gs[i, j])

        # Set labels
        if i == 0:
            ax.set_title("timeout" if c == "time_out" else c, fontsize=19)
        if j == 0:
            if g[0] is not None:
                ax.set_ylabel(g[1], fontsize=20)
            else:
                ax.set_ylabel(g[1], fontsize=20)

        # Remove chart borders and ticks
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # Get positions
        if g[0] is not None:
            row = generator.dataset.get_gen_class_rows(c)
        else:
            row = dataset.get_class_rows(c)

        tolerance = 0.05
        xpoints = row.filter(regex="^#x").values.flatten()
        ypoints = row.filter(regex="^#y").values.flatten()
        xpoints[np.isclose(xpoints, 0, atol=tolerance)] = 0
        ypoints[np.isclose(ypoints, 0, atol=tolerance)] = 0
        ids = ~(np.array(xpoints == 0) & np.array(ypoints == 0))
        xpoints = xpoints[ids]
        ypoints = ypoints[ids]
        # First array horiz. coords., second vertical
        # Player position heatmap
        # ax.hist2d(
        #     xpoints,
        #     ypoints,
        #     # bins=[np.arange(0, 1.0, 0.08), np.arange(0.0, 1.0, 0.08)],
        #     cmap="Greens",
        # )
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        nbins = 20
        k = kde.gaussian_kde([xpoints, ypoints])
        xi, yi = np.mgrid[
            0 : 1 : nbins * 1j,
            0 : 1 : nbins * 1j,
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        maxcnt = zi.max() if zi.max() > maxcnt else maxcnt
        # Make the plot
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="Greens")
        # Middle line
        ax.plot([0.5, 0.5], [0.0, 1.0], color="black")
        # Center circle
        ax.add_patch(Ellipse((0.5, 0.5), 0.2, 0.7, facecolor="none", ec="k", lw=2))
        # Areas
        ax.add_patch(
            Arc(
                (0.0, 0.5),
                0.2,
                0.9,
                angle=180,
                theta1=90,
                theta2=270,
                facecolor="none",
                ec="k",
                lw=2,
            )
        )
        ax.add_patch(
            Arc(
                (1.0, 0.5),
                0.2,
                0.9,
                theta1=90,
                theta2=270,
                facecolor="none",
                ec="k",
                lw=2,
            )
        )
        # Field
        ims.append(
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", ec="k", lw=2))
        )

        axs.append(ax)

        j += 1

    i += 1

cmap = mpl.cm.Greens
norm = mpl.colors.Normalize(vmin=0, vmax=round(maxcnt))
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs,
    orientation="vertical",
    fraction=0.1,
    aspect=60,
    pad=0.025,
)
cb.ax.tick_params(labelsize=20)
cb.outline.set_visible(False)

plt.savefig("figures/HeatmapPlaynet.pdf", format="pdf", bbox_inches="tight")
