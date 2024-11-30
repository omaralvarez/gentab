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

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Arc
from mycolorpy import colorlist as mcp
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
fig.subplots_adjust(wspace=0.15, hspace=0.30)

colors = mcp.gen_color(cmap="Greens", n=8)

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

        aspect_ratio = 0.38

        nbins = 35
        nbins_x = nbins
        nbins_y = int(nbins * aspect_ratio)

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = kde.gaussian_kde([xpoints, ypoints])
        xi, yi = np.mgrid[
            0 : 1 : nbins_x * 1j,
            0 : 1 : nbins_y * 1j,
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        maxcnt = zi.max() if zi.max() > maxcnt else maxcnt

        # Make the plot
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap="Greens")
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

        ax_histx = ax.inset_axes([0, 1.001, 1, 0.18], sharex=ax)
        ax_histy = ax.inset_axes([1.0001, 0, 0.08, 1], sharey=ax)
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        # Remove chart borders and ticks
        ax_histx.spines["top"].set_visible(False)
        ax_histx.spines["right"].set_visible(False)
        ax_histx.spines["bottom"].set_visible(False)
        ax_histx.spines["left"].set_visible(False)
        ax_histx.get_yaxis().set_ticks([])
        ax_histy.spines["top"].set_visible(False)
        ax_histy.spines["right"].set_visible(False)
        ax_histy.spines["bottom"].set_visible(False)
        ax_histy.spines["left"].set_visible(False)
        ax_histy.get_xaxis().set_ticks([])

        nbins_hist = 28
        nbins_x_hist = nbins_hist
        nbins_y_hist = int(nbins_hist * aspect_ratio)
        bins_x = np.linspace(0.0, 1.0, nbins_x_hist, endpoint=True)
        bins_y = np.linspace(0.0, 1.0, nbins_y_hist, endpoint=True)

        ax_histx.hist(xpoints, bins=bins_x, color=colors[5])
        ax_histy.hist(ypoints, bins=bins_y, orientation="horizontal", color=colors[5])

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
