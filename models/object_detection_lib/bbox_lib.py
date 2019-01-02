import tensorflow as tf

# EPSILON is in preventing numeric overflow when encoding bounding boxes into delta form.
EPSILON = 1e-6


def intersection(bbox1, bbox2):
    """Compute pairwise intersection areas between boxes.

    Args:
        bbox1: bbox holding N boxes
        bbox2: bbox holding M boxes

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=bbox1, num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=bbox2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(
        0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(
        0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def area(bbox):
    """Computes area of boxes.

    Args:
        bbox: bbox holding N boxes

    Returns:
        a tensor with shape [N] representing box areas.
    """
    y_min, x_min, y_max, x_max = tf.split(
        value=bbox, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def iou(bbox1, bbox2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        bbox1: bbox holding N boxes
        bbox2: bbox holding M boxes

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = intersection(bbox1, bbox2)
    areas1 = area(bbox1)
    areas2 = area(bbox2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def matched_iou(bbox1, bbox2):
    """Compute intersection-over-union between corresponding boxes in bboxs.

    Args:
        bbox1: bbox holding N boxes
        bbox2: bbox holding N boxes

    Returns:
        a tensor with shape [N] representing pairwise iou scores.
    """
    intersections = matched_intersection(bbox1, bbox2)
    areas1 = area(bbox1)
    areas2 = area(bbox2)
    unions = areas1 + areas2 - intersections
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def get_center_coordinates_and_sizes(box_corners):
    """Computes the center coordinates, height and width of the boxes.

    Args:
        box_corners: bounding box in corner representation.

    Returns:
        a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]


def encode_box_with_anchor(bbox, anchors):
    """Encode box with anchors

    Args:
        bbox: bounding box in corner representation.
        anchors: anchors in corner represnetation.

    Returns:
        bounding box in delta form.
    """
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ycenter, xcenter, h, w = get_center_coordinates_and_sizes(bbox)
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    return tf.transpose(tf.stack([ty, tx, th, tw]))


def decode_box_with_anchor(encoded_bbox, anchors):
    """Decode box with anchors.

    Args:
        encoded_bbox: bounding box in delta form.
        anchors: anchors in corner represnetation.

    Returns:
        Bounding box in corner representation.
    """
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ty, tx, th, tw = tf.unstack(tf.transpose(encoded_bbox))
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def normalizing_bbox(bbox, image_height, image_width):
    """Normalize bouding box with respect to image size.

    Args:
        bbox: bouding box in corner representation.
        image_height: image height.
        image_width: image width.

    Returns:
        Normalized bounding box coordinate to [0, 1].
    """
    if tf.rank(bbox).numpy() == 2:
        axis = 1
    elif tf.rank(bbox).numpy() == 3:
        axis = 2
    else:
        raise Exception(
            "bbox dimention has to be 2 or 3, but get {} instead".format(tf.rank(bbox)))

    y_min, x_min, y_max, x_max = tf.split(
        value=bbox, num_or_size_splits=4, axis=axis)

    normalized_bbox = tf.concat(
        [y_min / image_height, x_min / image_width,
            y_max / image_height, x_max / image_width],
        axis=axis)
    return normalized_bbox
