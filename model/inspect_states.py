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
    csv_path_m3 = Path("/Users/thomasbush/Downloads/features-states_m_3.csv")
    df_states = pd.read_csv(csv_path)
    df_states_m3 = pd.read_csv(csv_path_m3)
    return df_states, df_states_m3


@app.cell
def _(df_states):
    df_states.groupby("state")[['facing_angle_deg', 'head_speed', 'forepawR_tail_distance',
           'forepawL_tail_distance', 'head_acc', 'facing_angle_sin',
           'facing_angle_cos', 'ori_allBody_sin', 'ori_allBody_cos',
           'ori_head_sin', 'ori_head_cos']].median()
    return


@app.cell
def _(df_states, plt, sns):
    for feat in ['facing_angle_deg', 'head_speed', 'forepawR_tail_distance',
           'forepawL_tail_distance', 'head_acc', 'ori_allBody_sin', 'ori_allBody_cos',
           'ori_head_sin', 'ori_head_cos']:
        plt.figure(figsize=(6,4))
        sns.violinplot(data=df_states, x="state", y=feat, inner="box", cut=0)
        plt.title(f"{feat} by state")
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(df_states, plt, sns):
    state_colors = {
        0: "#1f77b4",  # blue
        1: "#ff7f0e",  # orange
        2: "#2ca02c",  # green
        3: "#d62728",  # red
        4: "#9467bd",  # purple
        5: "#8c564b",  # brown
    }

    sns.scatterplot(
        data=df_states,
        x="dist_head",
        y="facing_angle_deg",
        hue="state",
        palette=state_colors,
        s=10, alpha=0.6
    )
    plt.show()

    return (state_colors,)


@app.cell
def _(df_states, plt, sns, state_colors):
    sns.scatterplot(
        data=df_states, 
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
def _(df_states, plt, sns):
    sns.pairplot(
        df_states,
        vars=["dist_head", "facing_angle_deg"],
        hue="state",
        corner=True,
        plot_kws=dict(alpha=0.4, s=10),
    )
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
    return (feat_cols,)


@app.cell
def _(df_states, feat_cols, plt, sns, state_colors):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = df_states[feat_cols]
    Xz = StandardScaler().fit_transform(X)

    pc = PCA(n_components=2).fit_transform(Xz)
    df_states["pc1"] = pc[:,0]
    df_states["pc2"] = pc[:,1]

    sns.scatterplot(
        data=df_states,
        x="pc1",
        y="pc2",
        hue="state",
        s=3,
        alpha=0.5,
        palette=state_colors
    )
    # plt.savefig("/Users/thomasbush/Downloads/pca_states.png", dpi=150)
    plt.show()

    return PCA, StandardScaler


@app.cell
def _(PCA, StandardScaler, df_states, df_states_m3, feat_cols, pd):
    df_all = pd.concat([df_states, df_states_m3], ignore_index=True)

    X_all = df_all[feat_cols].values
    scaler = StandardScaler()

    X_all = df_all[feat_cols].values
    Xz_all = scaler.fit_transform(X_all)
    pca_all = PCA(n_components=3)
    pca_all.fit(Xz_all)

    print("Explained variance", pca_all.explained_variance_ratio_)
    return pca_all, scaler


@app.cell
def _(df_states, df_states_m3, feat_cols, pca_all, scaler):
    df2_pca = df_states.copy()
    X2 = scaler.transform(df_states[feat_cols])
    df2_pca[['pc1', 'pc2', 'pc3']] = pca_all.transform(X2)

    df3_pca = df_states_m3.copy()
    X3 = scaler.transform(df_states_m3[feat_cols])
    df3_pca[['pc1', 'pc2', 'pc3']] = pca_all.transform(X3)

    return df2_pca, df3_pca


@app.cell
def _(df2_pca, df3_pca, pd):
    df_all_pca = pd.concat([
        df2_pca.assign(mouse=2),
        df3_pca.assign(mouse=3)
    ])

    return


@app.cell
def _(df2_pca, plt, sns):
    sns.scatterplot(
        data=df2_pca, x="pc1", y="pc2",
        hue="state", palette="tab10",
        s=8, alpha=0.6
    )
    plt.title("Mouse 2 PCA / states")
    plt.show()
    return


@app.cell
def _(df3_pca, plt, sns):
    sns.scatterplot(
        data=df3_pca, x="pc1", y="pc2",
        hue="state", palette="tab10",
        s=8, alpha=0.6
    )
    plt.title("Mouse 3 PCA / states")
    plt.show()
    return


@app.cell
def _(df2_pca, df3_pca, plt, sns):
    sns.scatterplot(
        data=df2_pca.groupby("state")[["pc1", "pc2"]].mean().reset_index(),
        x="pc1", y="pc2", hue="state", palette="tab10",
        s=200, marker="o"
    )
    sns.scatterplot(
        data=df3_pca.groupby("state")[["pc1", "pc2"]].mean().reset_index(),
        x="pc1", y="pc2", hue="state", palette="tab10",
        s=200, marker="X", legend=False
    )
    plt.title("Mouse 2 (circles) vs Mouse 3 (X) state centroids")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
