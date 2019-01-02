import tarfile
import os
import json
import random

import tensorflow as tf
import numpy as np

"""Download data. """


def download_and_extract_data(tarfile_path, cam_num):
    output_dir = os.path.dirname(tarfile_path)
    if os.path.exists(os.path.join(output_dir, cam_num)):
        return output_dir
    tar = tarfile.open(tarfile_path)
    tar.extractall(output_dir)
    tar.close()
    return output_dir


"""Prepare data. Split and convert to tf records."""


def check_list(item):
    if not isinstance(item, list):
        item = [item]
    return item


def encode_float(item):
    item = check_list(item)
    return tf.train.Feature(float_list=tf.train.FloatList(value=item))


def encode_bytes(item):
    item = check_list(item)
    item = [it.encode('utf-8') for it in item]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=item))


def encode_int64(item):
    item = check_list(item)
    check_list(item)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=item))


def write_partition_tf(annotation, cam_data_dir, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for annot in annotation:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_name": encode_bytes(os.path.join(cam_data_dir, annot["image_name"])),
                        "mask_name": encode_bytes(os.path.join(cam_data_dir, annot["mask_name"])),
                        "bboxes": encode_float(np.array(annot["bboxes"]).reshape(-1).tolist()),
                        # -1 is to convert label [1, 10]. to [0, 9]
                        "labels": encode_int64([label - 1 for label in annot["labels"]])
                    }))
            writer.write(example.SerializeToString())


def partition_data(data_dir, cam_num, train_ratio):
    annotation = json.load(
        open(os.path.join(data_dir, cam_num, 'detection_annotation_converted.json'), 'r'))
    random.shuffle(annotation)
    train_len = int(len(annotation) * train_ratio)
    cam_data_dir = os.path.join(data_dir, cam_num)
    train_filepath = os.path.join(cam_data_dir, "train.tf")
    val_filepath = os.path.join(cam_data_dir, "val.tf")
    write_partition_tf(annotation[:train_len], data_dir, train_filepath)
    write_partition_tf(annotation[train_len:], data_dir, val_filepath)
    return train_filepath, val_filepath


def convert(tarfile_path):
    cam_num = "164"
    train_ratio = 0.8
    output_dir = download_and_extract_data(tarfile_path, cam_num="164")
    train_filepath, val_filepath = partition_data(
        output_dir, cam_num=cam_num, train_ratio=train_ratio)
    return train_filepath, val_filepath
