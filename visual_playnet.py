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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, Arc
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
                ax.set_ylabel(g[1], fontsize=18)
            else:
                ax.set_ylabel(g[1], fontsize=18)

        # Remove chart borders and ticks
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # Get positions
        if g[0] is not None:
            row = generator.dataset.get_random_gen_class_rows(c, 1)
        else:
            row = dataset.get_random_class_rows(c, 1)

        xpoints = row.filter(regex="^#x").values.flatten()
        ypoints = row.filter(regex="^#y").values.flatten()
        vxpoints = row.filter(regex="^#vx").values.flatten()
        vypoints = row.filter(regex="^#vy").values.flatten()
        ballx = row.filter(regex="^#ball_x").values.flatten()
        bally = row.filter(regex="^#ball_y").values.flatten()

        tolerance = 0.05
        xpoints[np.isclose(xpoints, 0, atol=tolerance)] = 0
        ypoints[np.isclose(ypoints, 0, atol=tolerance)] = 0
        vxpoints[np.isclose(vxpoints, 0, atol=tolerance)] = 0
        vypoints[np.isclose(vypoints, 0, atol=tolerance)] = 0
        ballx[np.isclose(ballx, 0, atol=tolerance)] = 0
        bally[np.isclose(bally, 0, atol=tolerance)] = 0
        ids = ~(
            np.array(xpoints == 0)
            & np.array(ypoints == 0)
            & np.array(vxpoints == 0)
            & np.array(vxpoints == 0)
        )
        xpoints = xpoints[ids]
        ypoints = ypoints[ids]
        vxpoints = vxpoints[ids]
        vypoints = vypoints[ids]
        ids = ~(np.array(ballx == 0) & np.array(bally == 0))
        ballx = ballx[ids]
        bally = bally[ids]

        # First array horiz. coords., second vertical
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
        # Player positions
        ax.plot(xpoints, ypoints, "o")
        # Ball
        # ax.plot(ballx, bally, "o", color="magenta")
        # Speeds
        ax.quiver(
            xpoints,
            ypoints,
            vxpoints,
            vypoints,
            color="orange",
            angles="uv",
            scale=4,
            headaxislength=3,
            headlength=3,
        )
        # Field
        ims.append(
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", ec="k", lw=2))
        )

        axs.append(ax)

        j += 1

    i += 1

plt.savefig("figures/VisualPlaynet.pdf", format="pdf", bbox_inches="tight")
