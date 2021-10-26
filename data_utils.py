from pathlib import Path
import re

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

import imageio


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


def descriptive_stats(df):

    dfs = []

    for col in [
        "label",
        "sex",
        "hand",
        "background",
        "illumination",
        "people_in_scene",
        "background_motion",
    ]:

        counts = df[col].value_counts(sort=False)
        counts.name = "n"

        as_per = counts / counts.sum()
        as_per.name = "%"

        _df = pd.concat([counts, as_per], axis=1)
        _df.index.name = col
        dfs.append(_df)

    return pd.concat(dfs, keys=[x.index.name for x in dfs])


def original_split_describe(df):
    """some descriptive stats of the original data split (found in metadata.csv)"""

    train = df[df["orig_set"] == "train"]
    test = df[df["orig_set"] == "test"]

    train_desc = descriptive_stats(train)
    test_desc = descriptive_stats(test)

    format_df = pd.concat([train_desc, test_desc], axis=1, keys=["train", "test"])
    format_df = format_df.replace(np.NaN, 0)

    # format_df.style.format("{:.2%}", subset=(format_df.columns.get_level_values(1) == "%"), na_rep=0)

    print(format_df.to_markdown(tablefmt="fancy_grid"))


def create_gif(path, img_sequence):
    imageio.mimsave(path, (img_sequence.numpy() * 255).astype(np.uint8), fps=30)