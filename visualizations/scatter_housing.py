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

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
from mycolorpy import colorlist as mcp
import geopandas as gpd

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

config = Config("configs/california_housing_cr.json")

labels = ["lowest", "lower", "low", "medium", "high", "higher", "highest"]
bins = [float("-inf"), 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, float("inf")]

dataset = Dataset(config, bins=bins, labels=labels)

colors = mcp.gen_color(cmap="viridis", n=len(dataset.class_names()))
cmap = mpl.cm.viridis

sizes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

gs = gridspec.GridSpec((len(gens) // 4) + 1, 4)
fig = plt.figure(figsize=(40, 30))
fig.subplots_adjust(wspace=0.08, hspace=0.25)

path = "us_state_info/tl_2023_us_state.shp"
df = gpd.read_file(path)
df = df.to_crs("EPSG:4326")
cali = df[df.STUSPS == "CA"]

axs, ims = [], []

i = 0
j = 0
for g in gens:
    if g[0] is not None:
        generator = g[0](dataset)
        generator.load_from_disk()

    ax = fig.add_subplot(gs[i, j])

    ax.set_xlim((-125, -113))
    ax.set_ylim((32, 43))

    ax.set_ylabel("Latitude", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=16)

    cali.boundary.plot(ax=ax, edgecolor="black", linewidth=2)

    x = []
    y = []
    k = 0
    for c in labels:
        # Get positions
        if g[0] is not None:
            if c != "highest":
                # rows = generator.dataset.get_gen_class_rows(c)
                rows = generator.dataset.get_random_gen_class_rows(c, 770)
            else:
                rows = generator.dataset.get_random_gen_class_rows(c, 300)
        else:
            if c != "highest":
                # rows = dataset.get_class_rows(c)
                rows = dataset.get_random_class_rows(c, 770)
            else:
                rows = dataset.get_random_class_rows(c, 300)

        lat = rows["Latitude"].values.flatten()
        lon = rows["Longitude"].values.flatten()

        # The scatter plot
        scat = ax.scatter(
            lon, lat, s=(k + 4) ** 2, c=colors[k], alpha=0.35, label=labels[k]
        )

        x = np.concatenate((x, lon), axis=None)
        y = np.concatenate((y, lat), axis=None)

        k += 1

    ax_histx = ax.inset_axes([0, 1.001, 1, 0.12], sharex=ax)
    ax_histy = ax.inset_axes([1.001, 0, 0.12, 1], sharey=ax)
    # No labels
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
    # Now determine nice limits by hand
    binwidth = 0.5
    xymax = 130
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color=colors[3])
    ax_histy.hist(y, bins=bins, orientation="horizontal", color=colors[3])

    ax_histx.set_title(g[1], fontsize=20, loc="right", y=1.0, pad=-20)

    axs.append(ax)

    if j == 3:
        j = 0
        if i == 3:
            break
        i += 1
    else:
        j += 1

ax = fig.add_subplot(gs[2, 3])

h, l = axs[0].get_legend_handles_labels()
ax.legend(
    h,
    l,
    loc="center",
    fontsize=35,
    markerscale=4,
    # title="Price",
    title_fontsize=38,
    frameon=False,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

plt.savefig("figures/VisualHousing.pdf", format="pdf", bbox_inches="tight")
