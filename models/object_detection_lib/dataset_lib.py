import functools

import tensorflow as tf

import anchor_lib
import bbox_lib
import matcher


def convert_to_dense(tensor):
    if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse_tensor_to_dense(tensor)
    return tensor


def parser(record):
    keys_to_features = {
        'image_name': tf.FixedLenFeature([], dtype=tf.string),
        'mask_name': tf.FixedLenFeature([], dtype=tf.string),
        'bboxes': tf.VarLenFeature(dtype=tf.float32),
        'labels': tf.VarLenFeature(dtype=tf.int64)
    }

    parsed = tf.parse_single_example(record, features=keys_to_features)
    parsed = {key: convert_to_dense(value) for key, value in parsed.iteritems()}

    image_name = parsed["image_name"]
    image_string = tf.read_file(image_name)
    image_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(image_string), tf.float32)

    mask_name = parsed["mask_name"]
    mask_string = tf.read_file(mask_name)
    mask_decoded = tf.image.decode_png(mask_string)
    # Reducing mask to two dimention.
    mask_decoded = tf.reduce_any(tf.cast(mask_decoded, tf.bool), -1)

    bboxes = tf.cast(parsed['bboxes'], tf.float32)
    bboxes = tf.reshape(bboxes, (-1, 4))

    labels = parsed["labels"]

    return {"image": image_decoded, "mask": mask_decoded, "bboxes": bboxes, "labels": labels}


def image_argumentation(image):
    image = tf.image.random_brightness(image, 0.125)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image


def preprocess_data(record, anchors, pos_iou_threshold, neg_iou_threshold,
                    neg_label_value, ignore_label_value, image_arg=False):

    if image_arg:
        record["image"] = image_argumentation(record["image"])

    # Wihten the dataset.
    record["image"] -= 0.5

    bboxes = record["bboxes"]
    labels = record["labels"]

    bboxes, labels = matcher.matching_bbox_label(
        bboxes, labels, anchors, pos_iou_threshold=pos_iou_threshold,
        neg_iou_threshold=neg_iou_threshold,
        neg_label_value=neg_label_value, ignore_label_value=ignore_label_value)

    mask = record["mask"]
    # ignore the indices outside of the mask.
    ay, ax, ah, aw = bbox_lib.get_center_coordinates_and_sizes(anchors)
    anchor_center_index = tf.cast(tf.transpose(tf.stack([ay, ax])), tf.int32)
    in_mask = tf.gather_nd(mask, anchor_center_index)
    ignore_labels = tf.ones_like(labels) * ignore_label_value
    labels = tf.where(in_mask, labels, ignore_labels)

    bboxes_encoded = bbox_lib.encode_box_with_anchor(bboxes, anchors)
    record["bboxes_preprocessed"] = bboxes_encoded
    record["labels_preprocessed"] = labels
    return record


def read_data(file_name, anchors, batch_size, pos_iou_threshold, neg_iou_threshold,
              neg_label_value, ignore_label_value, epoch=None, shuffle_buffer_size=None,
              image_arg=False):

    ds = tf.data.TFRecordDataset(file_name)
    ds = ds.repeat(epoch)
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.map(parser)

    preprocess_data_anchor = functools.partial(
        preprocess_data, anchors=anchors,
        pos_iou_threshold=pos_iou_threshold, neg_iou_threshold=neg_iou_threshold,
        neg_label_value=neg_label_value, ignore_label_value=ignore_label_value,
        image_arg=image_arg)

    ds = ds.map(preprocess_data_anchor)
    padded_shapes = {"bboxes": [None, 4], "labels": [None], "image": [None, None, 3],
                     "mask": [None, None],
                     "bboxes_preprocessed": [None, None], "labels_preprocessed": [None]}
    padding_values = {"bboxes": tf.constant(0, tf.float32),
                      "labels": tf.constant(ignore_label_value, tf.int64),
                      "image": tf.constant(0, tf.float32), "mask": tf.constant(False, tf.bool),
                      "bboxes_preprocessed": tf.constant(ignore_label_value, tf.float32),
                      "labels_preprocessed": tf.constant(0, tf.int64)}

    ds = ds.padded_batch(
        batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

    ds = ds.make_one_shot_iterator()

    return ds
