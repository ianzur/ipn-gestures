from pathlib import Path
import functools
import logging

import numpy as np
import pandas as pd
from tensorflow_datasets.core.features import video_feature
import imageio

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import adjust_brightness
import tensorflow_datasets as tfds

import ipn_hand.ipn_hand

# print(tf.config.list_physical_devices())

logger = tf.get_logger()
logger.setLevel(logging.INFO)

features={
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

def read_labelmap(path=None):

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


def decode_frame(serialized_image):
    """Decodes a single frame."""
    return tf.image.decode_jpeg(
        serialized_image,
        channels=ds_info.features["video"].shape[-1],
    )


def random_manipulation(example):
    """
    some data manipulation, 
    these should probably be implemented as layers in a preprocessing "model" 
    """

    video = example["video"]

    half = tf.constant(0.5)

    state = tf.random.uniform((2,)) #, minval=0, maxval=1, dtype=tf.dtypes.float32)

    flip_lr = state[0] > half
    flip_ud = state[1] > half
    brightness = tf.random.uniform((), minval=-0.5, maxval=0.5)
    quality = tf.random.uniform((), minval=20, maxval=100, dtype=tf.dtypes.int32)

    if flip_lr:
        video = tf.vectorized_map(
        tf.image.flip_left_right,
        video,
        # fn_output_signature=ds_info.features["video"].dtype,
    )

    if flip_ud:
        video = tf.vectorized_map(
        tf.image.flip_up_down,
        video,
        # fn_output_signature=ds_info.features["video"].dtype,
    )
  
    tf.debugging.assert_type(
        video, tf.dtypes.float32, message=None, name=None
    )
    video = tf.vectorized_map(
        functools.partial(tf.image.adjust_brightness, delta=brightness),
        video,
    )
  
    video = tf.map_fn(
        functools.partial(tf.image.adjust_jpeg_quality, jpeg_quality=quality),
        video,
        parallel_iterations=10
    )

    example["video"] = video
    
    return example


def decode_video(example, window_size, loop, start):
    """ 

    This can be called on a single example in eager execution,
    but was designed to be used with a tf.data.Dataset.map(...) call
    
    params:
        example: dict of Tensors
        window_size: int,
            how many frames do you want?
        start: str
            [start, random, centered], where to start sampling window from
        loop: bool (default=True)
            if window is bigger than n-Frames, loop img sequence to satisfy
    
    Notes:
        starts:
            - begin: at beginning of sequence
            - random: at a random frame
                - if loop required?: start = random((frames - window_size), frames))
                - else: start = random(0, (frames - window_size)), (only loop if required)
            - centered: center window in sequence
                - [center - window_size / 2, center + window_size / 2] 
    
    """

    video = example["video"]
    frames = tf.cast(example["frames"], dtype=tf.dtypes.int32)

    if start == "centered":
        raise NotImplementedError
        # start = frames - (window_size // 2)
        # pass

    elif start == "random":
        # tf.print("random")

        loops_required = window_size // frames
        if window_size == frames:
            loops_required = 0

        video = tf.repeat(video, [loops_required+1])

        sample_start = tf.random.uniform(
                (),
                minval=0,
                maxval=(frames*(loops_required+1) - window_size),
                dtype=tf.dtypes.int32
                )

        video = video[sample_start:sample_start+window_size]

    elif start == "start":
        # tf.print("start")

        if loop:
            loops_required = window_size // frames
            video = tf.repeat(video, [loops_required+1])
            video = video[0:window_size]
        else:
            video = video[0:frames]
              
    else:
        raise ValueError("please choose one of: start=[start, random, centered] ")

    # decode frames from jpeg to uint8 tensor
    video = tf.map_fn(
            decode_frame,
            video,
            fn_output_signature=ds_info.features["video"].dtype,
            parallel_iterations=10,
        )


    # convert to float tensor [0, 1] 
    video = tf.cast(video, tf.dtypes.float32) / 255.

    # pack converted tensor to example
    example["video"] = video

    return example


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


if __name__ == "__main__":

    # # Don't load video feature when creating df
    # ds, ds_info = tfds.load(
    #     "ipn_hand",
    #     data_dir="./data",
    #     split="train",  # currently there are no pre-defined train/val/test splits
    #     decoders=tfds.decode.PartialDecoding(features),  # do NOT read video data
    #     with_info=True,
    #     as_supervised=False,  # set True to only return (video, label) tuple
    # )

    # df = tfds2df(ds, ds_info)
    # original_split_describe(df)

    # print(df[df["frames"] <= 18]["label"].value_counts().to_markdown(tablefmt="grid"))

    # print(train_participants & test_participants)

    label_map = read_labelmap()

    # Don't load video feature when creating df
    ds, ds_info = tfds.load(
        "ipn_hand",
        data_dir="./data",
        split="train",  # currently there are no pre-defined train/val/test splits
        decoders={"video": tfds.decode.SkipDecoding()},  # skip decoding for now
        with_info=True,
        as_supervised=False,  # set True to only return (video, label) tuple
    )

    # ds_train = 
    # ds_validation = 
    # ds_test = 

    with tf.device("CPU"):
        ds = ds.map(functools.partial(decode_video, window_size=60, loop=True, start="random")).batch(10)

    i = 0
    ## Check the contents
    for item in ds:
        create_gif("./test.gif", item["video"][i])
        print(label_map[ds_info.features["label"].int2str(item["label"][i])])
        print(item["start"][i], item["end"][i])
        print(item["filename"][i])
        print(item["video"].shape)
        # print(item)
        break

