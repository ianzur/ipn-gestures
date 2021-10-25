"""ipn_hand dataset."""

from collections import namedtuple
import multiprocessing
from pathlib import Path
from typing import List, Tuple
import logging
import re

import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(ipn_hand): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """


https://gibranbenitez.github.io/IPN_Hand/
"""

# TODO(ipn_hand): BibTeX citation
_CITATION = """
"""

import pandas as pd
import numpy as np

logger = multiprocessing.get_logger()
logger.setLevel(logging.INFO)


def split_video(x):
    """return segments from video as list of tuples (key, np.array)"""
    # path: Path, slices: List[Tuple(int, int)]

    print(x)
    

    # all paths should be the same
    assert x["path"].nunique() == 1
    # print(x)

    path = x.iloc[0]["path"]

    cv2 = tfds.core.lazy_imports.cv2
    # # np = tfds.core.lazy_imports.numpy

    capture = cv2.VideoCapture(str(path))
    video_segments = []

    # TODO: check that all frames are labeled
    # x = x.sort_values(by="t_start")

    # assert all()
    match = re.search(pattern, x.iloc[0]["video"])
    vid_num = match.group("video_number")
    handedness = match.group("handedness")
    subject = match.group("subject")

    # i = 97
    # for _, slice in x.iterrows():
    #     start = slice["t_start"]
    #     end = slice["t_end"]

    #     frames = []
    #     for i in range(start, end + 1):
    #         ret, frame = capture.read()

    #         if not ret:
    #             print(f"Early exit: annotation suggests more frames exist in the video: {x.iloc[0]['video']} final_frame={i} vs. annotation={end}")
    #             break

    #         frames.append(frame)      

    #     video = np.stack(frames)

    #     video_segments.append((video, vid_num + chr(i), slice["label"], slice["video"], start, end, slice["frames"], handedness, subject))
    #     i += 1

    return video_segments


def read_annots_and_metas(path: Path):
    """read annotations and metadata, return as single dataframe
    
    Note:
        - columns names all lower cased
        - string labels from metadata are lowercased and spaces are removed

    """

    if not path.exists():
        raise RuntimeError(f"`{path}` not found")

    # read metadata
    df_meta = pd.read_table(path / "metadata.csv", delimiter=",", header=0, index_col=None)

    # clean and reformat metadata pre-merge
    df_meta = df_meta.rename(columns={"frames": "total_frames"})
    df_meta.columns = df_meta.columns.str.lower()
    df_meta.columns = df_meta.columns.str.replace(" ", "_")
    for col in ["sex", "hand", "background", "illumination", "people_in_scene", "background_motion"]:
        df_meta[col] = df_meta[col].str.lower()
        df_meta[col] = df_meta[col].str.strip(" ")

    # read annotations
    df_annot = pd.read_table(path / "Annot_List.txt", delimiter=",", header=0, index_col=None)

    # merge and drop now redundant "video_name" label
    df = pd.merge(df_annot, df_meta, left_on="video", right_on="video_name")
    df = df.drop(columns=["video_name"])

    # create "participant" label
    df["participant"] = df["video"].map(lambda x: "_".join(x.split("_")[:2]))

    # give each sequence (in the same video file) a unique ID
    df["unique_id"] = df.groupby("video", sort="t_start").cumcount()

    return df


def read_labels(path: Path):

    if not path.exists():
        raise RuntimeError(f"`{path}` is missing")

    df = pd.read_table(path, delimiter=",", header=0, index_col=None)

    return df


class IpnHand(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ipn_hand dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(ipn_hand): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "video": tfds.features.Video(
                        shape=(None, 240, 320, 3), dtype=tf.dtypes.uint8, encoding_format="jpeg"
                    ),
                    "label": tfds.features.ClassLabel(
                        names=read_labels(
                            Path("data/IPN_Hand/annotations/classIdx.txt")
                        )["label"]
                    ),
                    "start": tf.dtypes.uint32,
                    "end": tf.dtypes.uint32,
                    "frames": tf.dtypes.uint32,
                    "tot_frames": tf.dtypes.uint32,
                    "participant": tf.dtypes.string,
                    "sex": tfds.features.ClassLabel(names=["w", "m"]),
                    "hand": tfds.features.ClassLabel(names=["left", "right"]),
                    "background": tfds.features.ClassLabel(names=["clutter", "plain"]),
                    "illumination": tfds.features.ClassLabel(names=["stable", "light", "dark"]),
                    "people_in_scene": tfds.features.ClassLabel(names=['single', 'multi']),
                    "background_motion": tfds.features.ClassLabel(names=['static', 'dynamic']),
                    "orig_set": tfds.features.ClassLabel(names=['train', 'test']),
                    "filename": tf.dtypes.string
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("video", "label"),  # Set to `None` to disable
            homepage="https://gibranbenitez.github.io/IPN_Hand/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager=None):
        """Returns SplitGenerators."""

        # manual_download = Path.cwd() / "IPN_Hand"

        return {
            "train": self._generate_examples(Path.cwd() / "data" / "IPN_Hand"),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        # read annotations file
        df = read_annots_and_metas(path / "annotations" )
        # print(df.head())
        # df = df.drop(columns="id")
        # print(df.columns.tolist())

        frame_path = path / "frames"
               
        def _process_example(row):
            # print(f"{'*'*30} {row} {'*'*30}")
            frame_path = Path.cwd() / "data" / "IPN_Hand" / "frames"

            # video_list = list(frame_path / row[0] / row[0] + "_" + str(i).zfill(6) + ".jpg" for i in range(row[3], row[4]+1))

            video_list = []
            for i in range(row[3], row[4]+1):
                video_list.append(str(frame_path / row[0] / (row[0] + "_" + str(i).zfill(6) + ".jpg")))    

            key = row[0] + str(row[15])

            return key, {
                    'video': video_list,
                    'label': row[1],
                    'hand': row[8],
                    'participant': row[14],
                    'sex': row[7],
                    'background': row[9],
                    'illumination': row[10],
                    'people_in_scene': row[11],
                    'background_motion': row[12],
                    'orig_set': row[13], 
                    'start': row[3],
                    'end': row[4],
                    'frames': row[5],
                    'tot_frames': row[6],
                    'filename': row[0]
                }

        # this is slow, but not terribly slow
        for row in df.itertuples(index=False, name=None):
            yield _process_example(row)

        # TODO(ianzur): apacheBEAM, this segfaults on my machine
        # print(df.to_records(index=False))
        # return (
        #     beam.Create(df.to_records(index=False))
        #     | beam.Map(_process_example)
        # )


