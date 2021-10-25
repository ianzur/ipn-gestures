from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas.core import groupby

import tensorflow as tf
import tensorflow_datasets as tfds


def split_video(x):
    """return segments from video as list of tuples (key, np.array, label, id, handedness, subject, )"""
    # path: Path, slices: List[Tuple(int, int)]

    pattern = re.compile(r"(?P<id>\w+)_(?P<subject>\d+)_(?P<handedness>[R|L]{1})_#(?P<video_number>\d+)")

    # all paths should be the same
    assert x["path"].nunique() == 1
    # print(x)

    path = x.iloc[0]["path"]

    cv2 = tfds.core.lazy_imports.cv2
    # # np = tfds.core.lazy_imports.numpy

    capture = cv2.VideoCapture(str(path))
    video_segments = []

    # TODO: check that all frames are labeled
    x = x.sort_values(by="t_start")

    # assert all()
    match = re.search(pattern, x.iloc[0]["video"])
    vid_num = match.group("video_number")
    handedness = match.group("handedness")
    subject = match.group("subject")

    for _, slice in x.iterrows():
        start = slice["t_start"]
        end = slice["t_end"]

        frames = []
        for i in range(start, end + 1):
            ret, frame = capture.read()

            if not ret:
                continue
                # print(f"Early exit: annotation suggests more frames exist in the video: {x.iloc[0]['video']} final_frame={i} vs. annotation={end}")
                # break

            frames.append(frame)      

        video = np.stack(frames)

        video_segments.append((video, vid_num, slice["label"], slice["video"], start, end, slice["frames"], handedness, subject))
        i += 1

    return video_segments


def read_annot_and_meta(path: Path):

    if not path.exists():
        raise RuntimeError(f"`{path}` is missing")

    df_meta = pd.read_table(path / "metadata.csv", delimiter=",", header=0, index_col=None)
    df_meta.columns = df_meta.columns.str.lower()
    df_meta.columns = df_meta.columns.str.replace(" ", "_")

    for col in ["sex", "hand", "background", "illumination", "people_in_scene", "background_motion"]:
        df_meta[col] = df_meta[col].str.lower()
        df_meta[col] = df_meta[col].str.strip(" ")


    df_meta = df_meta.rename(columns={"frames": "total_frames"})

    df_annot = pd.read_table(path / "Annot_List.txt", delimiter=",", header=0, index_col=None)

    df = pd.merge(df_annot, df_meta, left_on="video", right_on="video_name")
    df = df.drop(columns=["video_name"])

    df["participant"] = df["video"].map(lambda x: "_".join(x.split("_")[:2]))

    # give each sequence (in the same video file) a unique ID
    df["unique_id"] = df.groupby("video", sort="t_start").cumcount()

    return df


def read_labels(path: Path):

    if not path.exists():
        raise RuntimeError(f"`{path}` is missing")

    df = pd.read_table(path, delimiter=",", header=0, index_col=None)

    return df

if __name__ == "__main__":
    df = read_annotations(Path("data/IPN_Hand/annotations"))
    print(df)

    for col in ["sex", "hand", "background", "illumination", "people_in_scene", "background_motion"]:
        print(f"{col}: {df[col].unique()}")


    # # split = [df.loc[value] for _, value in df.groupby(by="video").groups.items()]

    # for key, info in df.groupby(by="video"):
    #     split_video(info)

    # p = Path("/home/ian/Documents/projects/ipn-gestures/data/IPN_Hand/frames/1CM1_3_R_#226")

    # l = [int(p.name.split('.')[0].split("_")[-1]) for p in p.iterdir()]

    # s = pd.Series(l, dtype=int).sort_values().reset_index(drop=True)
    # print(s)
    # step = s[]
    
