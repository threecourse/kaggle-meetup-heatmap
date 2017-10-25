import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.externals import joblib
import matplotlib.pyplot as plt

np.random.seed(0)
sns.set()
from heatmap.heatmap2 import heatmap2

"""
https://www.kaggle.com/wendykan/lending-club-loan-data/data
https://www.kaggle.com/erykwalczak/initial-loan-book-analysis
"""

df = pd.read_csv("../input/loan.csv")
# columns = ["id", "annual_inc", "grade", "issue_d", "loan_amnt", "loan_status", "purpose"]
# df_small = df[columns].copy()
# joblib.dump(df_small, "../input/loan_small.pkl")
# df = joblib.load("../input/loan_small.pkl")

"""
# analyze

with open("../model/value_counts.txt", "w") as f:
    for c in df.columns:
        pr = lambda s: print(s, file=f)
        pr("----- {} -----".format(c))
        pr("")
        pr("isnull - value_counts")
        pr(df[c].isnull().value_counts())
        pr("")
        pr("unique counts - {}".format(len(df[c].unique())))
        pr("")
        pr("value_counts")
        pr(df[c].value_counts(dropna=False))
        pr("")

with open("../model/df_describe.txt", "w") as f:
    for c in df.columns:
        pr = lambda s: print(s, file=f)
        pr("----- {} -----".format(c))
        pr("")
        pr("-- describe --")
        pr(df[c].describe())
        pr("")

for c in df.columns:
    if(df[c].dtype.kind in list("iufc")):
        mask = df[c].notnull()
        sns.violinplot(df[c][mask])
        plt.show()

# cut extreme
print (df[df.isnull().any(axis=1)])
df = df[~df.isnull().any(axis=1)]

"""

# binning ---
def make_desc(bin):
    ret = []
    for i in range(len(bin)+1):
        d1 = "" if i == 0 else "{}k".format(round(bin[i-1] / 1000.0))
        d2 = "" if i == len(bin) else "{}k".format(round(bin[i] / 1000.0))
        ret.append("[{}] {}-{}".format(i, d1, d2))
    return np.array(ret)

annual_inc_bin = [20000.0, 40000.0, 60000.0, 80000.0, 100000.0, 150000.0, 200000.0]
annual_inc_bin_desc = make_desc(annual_inc_bin)
df["annual_inc_bin"] = annual_inc_bin_desc[np.digitize(df["annual_inc"], annual_inc_bin)]

loan_amnt_bin = [5000, 10000, 20000, 30000, 35000]
loan_amnt_bin_desc = make_desc(loan_amnt_bin)
df["loan_amnt_bin"] = loan_amnt_bin_desc[np.digitize(df["loan_amnt"], loan_amnt_bin)]

print (df["annual_inc_bin"].value_counts())
print (df["loan_amnt_bin"].value_counts())

# define bad_status ---
bad_status = ["Charged Off ", "Default",
              "Does not meet the credit policy. Status:Charged Off",
              "In Grace Period",
              "Default Receiver",
              "Late (16-30 days)", "Late (31-120 days)"]

df["bad_status"] = np.where(df["loan_status"].isin(bad_status), 1, 0)
print (df["bad_status"].value_counts())

def plot(index, columns):
    values = "bad_status"
    vmax = 0.10
    cellsize_vmax = 10000
    g_ratio = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    g_size = df.pivot_table(index=index, columns=columns, values=values, aggfunc="size")
    annot = np.vectorize(lambda x: "" if np.isnan(x) else "{:.1f}%".format(x * 100))(g_ratio)

    # adjust visual balance
    figsize = (g_ratio.shape[1] * 0.8, g_ratio.shape[0] * 0.8)
    cbar_width = 0.05 * 6.0 / figsize[0]

    f, ax = plt.subplots(1, 1, figsize=figsize)
    cbar_ax = f.add_axes([.91, 0.1, cbar_width, 0.8])
    heatmap2(g_ratio, ax=ax, cbar_ax=cbar_ax,
             vmax=vmax, cmap="PuRd", annot=annot, fmt="s", annot_kws={"fontsize": "small"},
             cellsize=g_size, cellsize_vmax=cellsize_vmax,
             square=True, ax_kws={"title": "{} x {}".format(index, columns)})

    plt.show()

plot("grade", "loan_amnt_bin")
plot("grade", "purpose")
plot("grade", "annual_inc_bin")
plot("loan_amnt_bin", "purpose")
plot("loan_amnt_bin", "annual_inc_bin")
plot("purpose", "annual_inc_bin")

