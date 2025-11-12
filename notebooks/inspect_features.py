import marimo

__generated_with = "0.17.7"
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
    return Path, pd, tqdm


@app.cell
def _(Path):
    features_dir = Path("/users/thomasbush/Downloads/shared WithTWB/")
    features_paths = [pathname for pathname in features_dir.iterdir()]
    return (features_paths,)


@app.cell
def _():
    # video path frames:
    import cv2


    # Custom function that returns total number of frames
    def count_frames_opencv(video_path):
        # Capturing the input video
        video = cv2.VideoCapture(video_path)

        # Accessing the CAP_PROP_FRAME_COUNT property
        # To get the total frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return total_frames

    # Example usage


    # Input video
    video_path = "/Users/thomasbush/Downloads/multicam_video_2025-05-07T12_16_20_cropped-v2_20250701121021/multicam_video_2025-05-07T12_16_20_central.mp4"

    # Calling the custom function
    frame_count = count_frames_opencv(video_path)
    print(f"Total frames: {frame_count}")
    return (frame_count,)


@app.cell
def _(features_paths, frame_count, pd, tqdm):
    # find the correct data:
    for pathname in tqdm(features_paths):
        #laod the pd
        df = pd.read_excel(pathname)
        #get length
        l = len(df)
        if l == frame_count:
            print("Correct feature path")
            print(pathname.name)
        
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
