import tensorflow as tf
import bbox_lib


def hard_negative_loss_mining(c_loss, negative_mask, k):
    """Hard negative mining in classification loss."""
    # make sure at least one negative example
    k = tf.maximum(k, 1)
    # make sure at most all negative.
    k = tf.minimum(k, c_loss.shape[-1])
    neg_c_loss = c_loss * negative_mask
    neg_c_loss = tf.nn.top_k(neg_c_loss, k)[0]
    return tf.reduce_sum(neg_c_loss)


def compute_loss(network_output, bboxes, labels, num_classes, c_weight, r_weight,
                 neg_label_value, ignore_label_value, negative_ratio):
    """Compute loss function."""

    batch_size = bboxes.shape[0].value
    one_hot_labels = tf.one_hot(labels + 1, num_classes + 1)
    negative_mask = tf.cast(tf.equal(labels, neg_label_value), tf.float32)
    positive_mask = tf.cast(tf.logical_and(tf.not_equal(labels, ignore_label_value),
                                           tf.not_equal(labels, neg_label_value)), tf.float32)

    classification_output = network_output[0]
    classification_output = tf.reshape(
        classification_output, [batch_size, -1, num_classes + 1])

    c_loss = tf.losses.softmax_cross_entropy(
        one_hot_labels, classification_output, reduction=tf.losses.Reduction.NONE)

    num_positive = tf.cast(tf.reduce_sum(positive_mask), tf.int32)
    pos_c_loss = tf.reduce_sum(c_loss * positive_mask)
    neg_c_loss = hard_negative_loss_mining(c_loss, negative_mask,
                                           num_positive * negative_ratio)

    c_loss = (pos_c_loss + neg_c_loss) / batch_size

    regression_output = network_output[1]
    regression_output = tf.reshape(
        regression_output, [batch_size, -1, 4])
    r_loss = tf.losses.huber_loss(regression_output, bboxes, delta=1,
                                  reduction=tf.losses.Reduction.NONE)

    r_loss = tf.reduce_sum(
        r_loss * positive_mask[..., tf.newaxis]) / batch_size

    return c_weight * c_loss + r_weight * r_loss


def predict(network_output, mask, score_threshold, neg_label_value, anchors,
            max_prediction, num_classes):
    """Decode predictions from the neural network."""

    classification_output = network_output[0]
    batch_size, _, _, output_dim = classification_output.get_shape().as_list()
    regression_output = network_output[1]
    bbox_list = []
    label_list = []

    ay, ax, ah, aw = bbox_lib.get_center_coordinates_and_sizes(anchors)
    anchor_center_index = tf.cast(tf.transpose(tf.stack([ay, ax])), tf.int32)
    for single_classification_output, single_regression_output, single_mask in zip(
            classification_output, regression_output, mask):
        # num_classes + 1 due to the negative class.
        single_classification_output = tf.reshape(
            single_classification_output, [-1, num_classes + 1])
        single_classification_output = tf.nn.softmax(
            single_classification_output, -1)

        max_confidence = tf.reduce_max(single_classification_output, -1)
        confident_mask = max_confidence > score_threshold
        # - 1 due to the negative class.
        max_index = tf.argmax(single_classification_output, 1) - 1
        non_negative_mask = tf.not_equal(max_index, -1)
        in_mask = tf.gather_nd(single_mask, anchor_center_index)
        foreground_mask = tf.logical_and(
            in_mask, tf.logical_and(confident_mask, non_negative_mask))

        valid_labels = tf.boolean_mask(max_index, foreground_mask)

        single_regression_output = tf.reshape(single_regression_output, [-1, 4])
        predicted_bbox = bbox_lib.decode_box_with_anchor(
            single_regression_output, anchors)
        valid_boxes = tf.boolean_mask(predicted_bbox, foreground_mask)
        valid_confidence_score = tf.boolean_mask(
            max_confidence, foreground_mask)

        selected_indices = tf.image.non_max_suppression(
            valid_boxes, valid_confidence_score, max_prediction)

        valid_boxes = tf.gather(valid_boxes, selected_indices)
        valid_labels = tf.gather(valid_labels, selected_indices)
        bbox_list.append(valid_boxes)
        label_list.append(valid_labels)

    return bbox_list, label_list
