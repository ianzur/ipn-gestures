from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

import ipn_hand.ipn_hand


# x = ipn_hand.ipn_hand.IpnHand()
# x._split_generators(None)

# p = Path("./data/IPN_Hand")

# a = ipn_hand.ipn_hand.read_labels(Path("data/IPN_Hand/annotations/classIdx.txt"))
# print(a.head())


# a = ipn_hand.ipn_hand.read_annotations(Path("data/IPN_Hand/annotations/Annot_List.txt"))
# print(a['video'].unique())

# a["path"] = p / "videos" / ( a['video'] + ".avi" )

ds, ds_info = tfds.load(
        "ipn_hand",
        data_dir="./data",
        split="train",  # currently there are no pre-defined train/val/test splits
        with_info=True,
        as_supervised=False,  # set False to return participant & attempt numbers, in addition to defined (features, gesture) tuple
    )




