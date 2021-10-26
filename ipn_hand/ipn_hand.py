"""ipn_hand dataset."""

from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# import numpy as np


_DESCRIPTION = """
The IPN Hand dataset contains more than 4,000 gesture instances and 800,000 frames from 50 subjects.
We design 13 static and dynamic gestures for interaction with touchless screens. 
Compared to other publicly available hand gesture datasets, IPN Hand includes the largest number of 
continuous gestures per video, and the largest speed of intra-class variation.

The data collection was designed considering real-world issues of continuous HGR, 
including continuous gestures performed without transitional states, natural movements as non-gesture segments, 
scenes including clutter backgrounds, extreme illumination conditions, as well as static and dynamic environments.
"""

_CITATION = """
@inproceedings{bega2020IPNhand,
  title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
  author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
  booktitle={25th International Conference on Pattern Recognition, {ICPR 2020}, Milan, Italy, Jan 10--15, 2021},
  pages={4340--4347},
  year={2021},
  organization={IEEE}
}
"""

_MANUAL_DOWNLOAD_INSTRUCTIONS = """
https://gibranbenitez.github.io/IPN_Hand/ click download link. 

Download and extract `frames/frames0X.tgz` to folder:
`ipn-gestures/data/IPN_Hand/frames/<vid_name>/<vid_name>_00XXXX.jpg`

And and `annotations/*` to folder:
`ipn-gestures/data/IPN_Hand/annotations/*`

e.g.
```
data/IPN_Hand/
├── annotations
│   ├── Annot_List.txt
│   ├── classIdx.txt
│   ├── metadata.csv
│   ├── ...
├── frames
│   ├── 1CM1_1_R_#217
│   │   ├── *000001.jpg
│   │   ├── *000002.jpg
│   │   ├── *000003.jpg
│   │   ├── ...
│   ├── 1CM1_1_R_#218
│   ├── 1CM1_1_R_#219
│   ├── ...
```
"""

# TODO: resolve IPN-hand issue #11 before attempting to use video data to create tfrecords
# def split_video(x):
#     """return segments from video as list of tuples (key, np.array)"""
#     # path: Path, slices: List[Tuple(int, int)]

#     print(x)
    

#     # all paths should be the same
#     assert x["path"].nunique() == 1
#     # print(x)

#     path = x.iloc[0]["path"]

#     cv2 = tfds.core.lazy_imports.cv2
#     # # np = tfds.core.lazy_imports.numpy

#     capture = cv2.VideoCapture(str(path))
#     video_segments = []

#     # TODO: check that all frames are labeled
#     # x = x.sort_values(by="t_start")

#     # assert all()
#     match = re.search(pattern, x.iloc[0]["video"])
#     vid_num = match.group("video_number")
#     handedness = match.group("handedness")
#     subject = match.group("subject")

#     # i = 97
#     # for _, slice in x.iterrows():
#     #     start = slice["t_start"]
#     #     end = slice["t_end"]

#     #     frames = []
#     #     for i in range(start, end + 1):
#     #         ret, frame = capture.read()

#     #         if not ret:
#     #             print(f"Early exit: annotation suggests more frames exist in the video: {x.iloc[0]['video']} final_frame={i} vs. annotation={end}")
#     #             break

#     #         frames.append(frame)      

#     #     video = np.stack(frames)

#     #     video_segments.append((video, vid_num + chr(i), slice["label"], slice["video"], start, end, slice["frames"], handedness, subject))
#     #     i += 1

#     return video_segments


def read_annots_and_metas(path: Path):
    """read annotations and metadata, return as single dataframe
    
    Note:
        - columns names all lower cased
        - string labels from metadata are lowercased and spaces are removed

    """

    if not path.exists():
        raise RuntimeError(_MANUAL_DOWNLOAD_INSTRUCTIONS)

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
        raise RuntimeError(_MANUAL_DOWNLOAD_INSTRUCTIONS)

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
            supervised_keys=("video", "label"),
            homepage="https://gibranbenitez.github.io/IPN_Hand/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager=None):
        """Returns SplitGenerators."""

        path = Path.cwd() / "data" / "IPN_Hand"

        if not path.exists():
            raise RuntimeError(_MANUAL_DOWNLOAD_INSTRUCTIONS)

        return {
            "train": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        # read annotations file
        df = read_annots_and_metas(path / "annotations" )

        frame_path = path / "frames"
               
        def _process_example(row):

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


