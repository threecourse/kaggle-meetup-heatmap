import numpy as np; np.random.seed(0)
import seaborn as sns;
sns.set()
import matplotlib.pyplot as plt
from heatmap.heatmap2 import heatmap2
import pandas as pd

def heatmap2_test1():
    f, axes = plt.subplots(2, 2, figsize=(12, 8))

    flights = sns.load_dataset("flights")
    flights = flights.pivot("month", "year", "passengers")
    cellsize = np.linspace(0.0, 10.0, flights.size).reshape(flights.shape)

    sns.heatmap(flights, ax=axes[0, 0], square=True)
    axes[0, 0].set_title("original seaborn heatmap")

    heatmap2(flights, ax=axes[0, 1], square=True, ax_kws={"title": "heatmap2(same cell sizes)"})
    heatmap2(flights, ax=axes[1, 0], square=True, cellsize=cellsize,
             ax_kws={"title": "heatmap2(change cell sizes)"})
    heatmap2(flights, ax=axes[1, 1], square=True, cellsize=cellsize,
             ax_kws={"title": "heatmap2(with annotation)"},
             annot=True, fmt="d", annot_kws={"fontsize": "x-small"})

    plt.tight_layout()
    plt.show()

def heatmap2_test2():
    f, axes = plt.subplots(2, 2, figsize=(12, 8))

    n = 10
    data_val = np.random.rand(n, n) - 0.5
    data = pd.DataFrame(data_val, index=list("abcdefghij"), columns=list("ABCDEFGHIJ"))
    cellsize = np.random.rand(n, n)
    annot = np.array(list("ox-"))[np.random.choice(range(3), (n,n))]
    mask = annot == "-"
    cmap = sns.color_palette("RdBu")

    heatmap2(data, ax=axes[0, 0], cmap=cmap, square=True, cellsize=cellsize, annot=annot, fmt="s",
             ax_kws={"title": "different colormap"},)
    heatmap2(data, ax=axes[0, 1], cmap=cmap, square=True, mask=mask, cellsize=cellsize, annot=annot, fmt="s",
             ax_kws={"title": "apply mask"}, )
    heatmap2(data, ax=axes[1, 0], cmap=cmap, square=True, cellsize=cellsize,
             rect_kws={"edgecolor": "black", "linewidth": 2.0},
             ax_kws={"title": "apply rect keyword arguments"}, )
    heatmap2(data, ax=axes[1, 1], cmap=cmap, square=True, cellsize=cellsize,
             ax_kws={"title": "apply ax keyword arguments", "facecolor": "dimgray"})

    plt.tight_layout()
    plt.show()

heatmap2_test1()
heatmap2_test2()