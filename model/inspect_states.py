import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import seaborn as sns
    return Path, pd, plt, sns


@app.cell
def _(Path, pd):
    csv_path = Path("/users/thomasbush/Downloads/features-states_m_2.csv")
    csv_path_m3 = Path("/users/thomasbush/Downloads/features-states.csv")
    df_states = pd.read_csv(csv_path)
    df_states_m3 = pd.read_csv(csv_path_m3)
    return df_states, df_states_m3


@app.cell
def _(df_states_m3):
    df_states_m3.groupby("state")[['facing_angle_deg', 'head_speed', 'forepawR_tail_distance',
           'forepawL_tail_distance', 'head_acc', 'facing_angle_sin',
           'facing_angle_cos', 'ori_allBody_sin', 'ori_allBody_cos',
           'ori_head_sin', 'ori_head_cos']].median()
    return


@app.cell
def _(df_states_m3):
    df_states_m3.columns
    return


@app.cell
def _(df_states_m3, plt, sns):
    for feat in ['dist_head',
           'facing_angle_deg', 'head_speed', 'facing_angle_sin',
           'facing_angle_cos']:
        plt.figure(figsize=(6,4))
        sns.violinplot(data=df_states_m3, x="state", y=feat, inner="box", cut=0)
        plt.title(f"{feat} by state")
        plt.tight_layout()
        plt.savefig(f"/Users/thomasbush/Downloads/{feat}_m3_hmm.png")
        plt.show()
    return


@app.cell
def _(df_states_m3, plt, sns):
    state_colors = {
        0: "#1f77b4",  # blue
        1: "#ff7f0e",  # orange
        2: "#2ca02c",  # green
        3: "#d62728",  # red
        4: "#9467bd",  # purple
        5: "#8c564b",  # brown
    }

    sns.scatterplot(
        data=df_states_m3,
        x="dist_head",
        y="facing_angle_deg",
        hue="state",
        palette=state_colors,
        s=10, alpha=0.6
    )
    plt.show()
    return (state_colors,)


@app.cell
def _(df_states_m3, plt, sns, state_colors):
    sns.scatterplot(
        data=df_states_m3, 
        x="dist_head", 
        y="head_speed", 
        hue="state",
        s=10, alpha=0.5,
        palette=state_colors
    )
    plt.ylim(0, 100)
    plt.show()
    return


@app.cell
def _(df_states_m3, plt, sns):
    sns.pairplot(
        df_states_m3,
        vars=["dist_head", "facing_angle_deg", "head_speed"],
        hue="state",
        corner=True,
        plot_kws=dict(alpha=0.4, s=10),
    )
    plt.savefig("/Users/thomasbush/Downloads/pairplot.png")
    plt.show()
    return


@app.cell
def _(df_states, plt, sns):
    feat_cols = ['facing_angle_deg', 'head_speed', 'forepawR_tail_distance', 'forepawL_tail_distance', 'head_acc', 'ori_allBody_sin', 'ori_allBody_cos','ori_head_sin', 'ori_head_cos']
    for state in df_states["state"].unique():
        subset = df_states[df_states["state"] == state]

        plt.figure(figsize=(10,4))
        sns.boxplot(data=subset[feat_cols])
        plt.title(f"State {state} – Feature Distributions")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(df_states_m3):
    df_states_m3.columns
    return


@app.cell
def _(df_states_m3, plt, sns, state_colors):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = df_states_m3[["dist_head", "facing_angle_cos", "head_speed", "facing_angle_sin"]]
    Xz = StandardScaler().fit_transform(X)

    pc = PCA(n_components=2).fit_transform(Xz)
    df_states_m3["pc1"] = pc[:,0]
    df_states_m3["pc2"] = pc[:,1]

    sns.scatterplot(
        data=df_states_m3,
        x="pc1",
        y="pc2",
        hue="state",
        s=3,
        alpha=0.5,
        palette=state_colors
    )
    # plt.savefig("/Users/thomasbush/Downloads/pca_states.png", dpi=150)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
