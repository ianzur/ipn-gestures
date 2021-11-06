import functools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import data_utils

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import ipn_hand.ipn_hand

RGB = 3
GRAY = 1

def decode_frame(serialized_image):
    """Decodes a single frame."""
    return tf.image.decode_jpeg(
        serialized_image,
        channels=GRAY # RGB | GRAY
    )

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
    frames = tf.cast(example["frames"], dtype=tf.dtypes.int32) #/ 10

    # TODO: investigate sampling every nth frame (sequential frames are practically the same.)
    # video = video[::10]

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
        raise ValueError("please choose one of: start=[start, random, centered]")

    # decode frames from jpeg to uint8 tensor
    video = tf.map_fn(
            decode_frame,
            video,
            fn_output_signature=ds_info.features["video"].dtype,
            parallel_iterations=10,
        )

    video = tf.vectorized_map(
        functools.partial(tf.image.resize, size=[120, 160]),
        video,
    )

    # convert to float tensor [0, 1] 
    video = tf.cast(video, tf.dtypes.float32) / 255.

    # pack converted tensor to example
    example["video"] = video

    return example


def one_hot(example):
    label = example["label"]
    label = tf.one_hot(label, depth=18)
    example["label"] = label
    return example


def build_model(time=60, height=120, width=160, depth=1):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((time, height, width, depth))

    #inputs = layers.Masking()(inputs)

    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=18, activation="sigmoid")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name="3dcnn")
    return model


from tensorflow import keras
from tensorflow.keras import layers


def get_model2(seq_length=200, width=128, height=128, depth=3):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((seq_length, height, width, depth))

    def cnn():
        cnn = keras.Sequential()
        cnn.add(layers.Conv2D(filters=16, kernel_size=3, activation="relu"))
        cnn.add(layers.MaxPool2D(pool_size=3))
        cnn.add(layers.BatchNormalization())
        cnn.add(layers.Conv2D(filters=16, kernel_size=3, activation="relu"))
        cnn.add(layers.MaxPool2D(pool_size=3))
        cnn.add(layers.BatchNormalization())
        cnn.add(layers.Conv2D(filters=16, kernel_size=3, activation="relu"))
        cnn.add(layers.MaxPool2D(pool_size=3))
        cnn.add(layers.BatchNormalization())
        # cnn.add(layers.Conv2D(filters=16, kernel_size=3, activation="relu"))
        # cnn.add(layers.MaxPool2D(pool_size=3))
        # cnn.add(layers.BatchNormalization())
        #cnn.add(layers.GlobalAveragePooling2D())
        cnn.add(layers.Flatten())
        return cnn
    x = layers.TimeDistributed(cnn())(inputs)
    
    x = layers.LSTM(512, activation='tanh')(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=18, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


if __name__ == "__main__":
    ds, ds_info = tfds.load(name='ipn_hand', data_dir='./data', as_supervised=False, decoders={"video": tfds.decode.SkipDecoding()}, split='train', with_info=True)

    window = 32

    ds_train, ds_val, ds_test = data_utils.split(ds)

    # decode video & resize
    ds_train = ds_train.map(functools.partial(decode_video, window_size=window, loop=True, start="start"), num_parallel_calls=tf.data.AUTOTUNE).batch(16)
    ds_val = ds_val.map(functools.partial(decode_video, window_size=window, loop=True, start="start"), num_parallel_calls=tf.data.AUTOTUNE).batch(16)
    ds_test = ds_test.map(functools.partial(decode_video, window_size=window, loop=True, start="start"), num_parallel_calls=tf.data.AUTOTUNE).batch(16)

    # one hot label
    ds_train = ds_train.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.map(lambda x: (x["video"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(lambda x: (x["video"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(lambda x: (x["video"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)

    # i = 0
    # for item in ds_val:
    #     data_utils.create_gif("./test3.gif", item[i][0])
    #     # print(label_map[ds_info.features["label"].int2str(item["label"][i])])
    #     # print(item["start"][i], item["end"][i])
    #     # print(item["filename"][i])
    #     # print(item["video"].shape)
    #     # print(item)
    #     break

    # Build model.
    model = get_model2(seq_length=window, height=120, width=160, depth=GRAY)
    model.summary(line_length=100)

    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3), tf.keras.callbacks.TensorBoard()]
    model.fit(ds_train, validation_data=ds_val, epochs=20, callbacks=callbacks)