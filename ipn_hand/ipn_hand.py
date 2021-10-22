"""ipn_hand dataset."""

import multiprocessing
from pathlib import Path
from typing import List, Tuple
import logging

import tensorflow_datasets as tfds
import tensorflow as tf

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

  
  # logger.info(x)
  print(x)
  path, id, label, start, end, frames = x[0], x[1], x[2], x[3], x[4], x[5]
  print(path, id, label, start, end, frames)

  # print(f"{path}, [{','.join([f'{i[0]} : {i[1]}' for i in slices]) }]")

  cv2 = tfds.core.lazy_imports.cv2
  # # np = tfds.core.lazy_imports.numpy

  videos = []
  for slice in slices:
    start, end = slice

    video = cv2.VideoCapture(path)

    frames = []
    for i in range(start, end + 1):
      frames.append(video.read())

    video = np.stack(frames)
    print(video.shape)

    videos.append(video)

  return videos
    
  
def read_annotations(path: Path):

  if not path.exists():
    raise RuntimeError(f"`{path}` is missing")

  # pd = tfds.core.lazy_import.pandas

  df = pd.read_table(path, delimiter=",", header=0, index_col=None)

  return df

def read_labels(path: Path):
    
    if not path.exists():
      raise RuntimeError(f"`{path}` is missing")

    df = pd.read_table(path, delimiter=",", header=0, index_col=None)

    return df


class IpnHand(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ipn_hand dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(ipn_hand): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'video': tfds.features.Video(shape=(None, None, None, 3), dtype=tf.dtypes.uint8),
            'label': tfds.features.ClassLabel(names=read_labels(Path("data/IPN_Hand/annotations/classIdx.txt"))['label']),
            't_start': tf.dtypes.uint32,
            't_end': tf.dtypes.uint32,
            'frames': tf.dtypes.uint32,
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://gibranbenitez.github.io/IPN_Hand/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    
    # TODO(ipn_hand): Returns the Dict[split names, Iterator[Key, Example]]

    # manual_download = Path.cwd() / "IPN_Hand" 

    return {
        'train': self._generate_examples( Path.cwd() / "data" / "IPN_Hand" ),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(ipn_hand): Yields (key, example) tuples from the dataset
    print(path)

    # read annotations file
    annotations = read_annotations( path / "annotations" / "Annot_List.txt" )

    annotations["path"] = path / "videos" / ( annotations['video'] + ".avi" )

    list_o_tuples = list(annotations.groupby(by=["video"])[["t_start", "t_end"]].agg(list))

    # from multiprocessing import Pool

    # with Pool(1) as pool:

    #   for videos in pool.imap_unordered(split_video, list_o_tuples):

    for tup in list_o_tuples:

      print(tup)

      
      for video in split_video(tup):
        
        print(video)

          # label = 
          # key = 

          # yield key, {
          #     'video': video,
          #     'label': label,
          # }
    
