from pathlib import Path
import re

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

import imageio


# features of dataset (use with tfds.load(...,  decoders=tfds.decode.PartialDecoding(features), ... ))
# to ignore video sequence and load metadata
META_FEATURES={
    "video": False,
    "label": True,
    "start": True,
    "end": True,
    "frames": True,
    "tot_frames": True,
    "participant": True,
    "sex": True,
    "hand": True,
    "background": True,
    "illumination": True,
    "people_in_scene": True,
    "background_motion": True,
    "orig_set": True,
    "filename": True
}

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
    """save image sequence as gif"""
    imageio.mimsave(path, (img_sequence.numpy() * 255).astype(np.uint8), fps=30)


def read_labelmap(path=None):
    """returns as dictionary {'D0X': 'no-gesture', ...}"""

    if path is None:
        path = Path("./ipn_hand/class_labelmap.csv")

    return pd.read_table(
        path, sep=",", index_col=[0], header=None, squeeze=True
    ).to_dict()


def tfds2df(ds, ds_info):
    """return dataset as dataframe (see: warning)

    Warning:
        - ** do NOT use `tfds.as_dataframe(...)` without ignoring video feature **
            > this will attempt to load all video sequences into your RAM
        - or you can "take" a subset of the ds object `ds.take(2)`
    """

    df = tfds.as_dataframe(ds, ds_info=ds_info)
    print(df.columns)

    # decode features
    for feature in [
        "label",
        "sex",
        "hand",
        "background",
        "illumination",
        "people_in_scene",
        "background_motion",
        "orig_set",
    ]:
        df[feature] = df[feature].map(ds_info.features[feature].int2str)

    # map label to human readable
    df["label"] = df["label"].map(read_labelmap())

    # decode participant names
    df["participant"] = df["participant"].str.decode("utf-8")

    return df
