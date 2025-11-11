import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    from pathlib import Path
    from tqdm import tqdm
    import pandas as pd
    import openpyxl
    return Path, pd, xr


@app.cell
def _(Path):
    features_dir = Path("/users/thomasbush/Downloads/shared WithTWB/")
    features_paths = [pathname for pathname in features_dir.iterdir()]
    return (features_paths,)


@app.cell
def _(features_paths, pd, xr):
    df_feature_1 = pd.read_excel(features_paths[0])
    df = df_feature_1.copy()
    df.index.name = "frame"   # important for xarray dims naming

    ds = xr.Dataset.from_dataframe(df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
