import tensorflow as tf


def expanded_shape(orig_shape, start_dim, num_dims):
    """Inserts multiple ones into a shape vector.

    Inserts an all-1 vector of length num_dims at position start_dim into a shape.
    Can be combined with tf.reshape to generalize tf.expand_dims.

    Args:
        orig_shape: the shape into which the all-1 vector is added (int32 vector)
        start_dim: insertion position (int scalar)
        num_dims: length of the inserted all-1 vector (int scalar)

    Returns:
        An int32 vector of length tf.size(orig_shape) + num_dims.
    """
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape


def anchor_gen(grid_height, grid_width, scales=(0.5, 1.0, 2.0), aspect_ratios=(0.5, 1.0, 2.0),
               base_anchor_size=[32.0, 32.0], anchor_stride=[16.0, 16.0],
               anchor_offset=[0.0, 0.0]):
    """ Generate anchor.

    Args:
        grid_height: (int). The height of the grid.
        grid_width: (int). The width of the grid.
        scales: (list of float). The scales of the anchor.
        aspect_ratios: (list of float). The ratios of anchor.
        base_anchor_size: (list of float). The base anchor size.
        anchor_stride: (list of float). The stride of anchor.
        anchor_offset: (list of float). The offset of anchor.
      x: A tensor of arbitrary shape and rank. xgrid will contain these values
         varying in its last dimensions.
      y: A tensor of arbitrary shape and rank. ygrid will contain these values
         varying in its first dimensions.

    Return:
        anchors: (2D Tensors). Generated anchors in corner representation.
    """
    base_anchor_size = tf.to_float(tf.convert_to_tensor(base_anchor_size))
    anchor_stride = tf.to_float(tf.convert_to_tensor(anchor_stride))
    anchor_offset = tf.to_float(tf.convert_to_tensor(anchor_offset))

    scales_grid, aspect_ratios_grid = meshgrid(scales, aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
    anchors = tile_anchors(grid_height,
                           grid_width,
                           scales_grid,
                           aspect_ratios_grid,
                           base_anchor_size,
                           anchor_stride,
                           anchor_offset)

    return anchors


def meshgrid(x, y):
    """Tiles the contents of x and y into a pair of grids.

    Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
    are vectors. Generally, this will give:
    xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
    ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)
    Keep in mind that the order of the arguments and outputs is reverse relative
    to the order of the indices they go into, done for compatibility with numpy.
    The output tensors have the same shapes.  Specifically:
    xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
    ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())

    Args:
        x: A tensor of arbitrary shape and rank. xgrid will contain these values
           varying in its last dimensions.
        y: A tensor of arbitrary shape and rank. ygrid will contain these values
           varying in its first dimensions.

    Returns:
        A tuple of tensors (xgrid, ygrid).
    """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))
    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid


def tile_anchors(grid_height, grid_width, scales, aspect_ratios, base_anchor_size,
                 anchor_stride, anchor_offset):
    """ Tile the anchors. """

    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    # Get a grid of box centers
    y_centers = tf.to_float(tf.range(grid_height))
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = tf.to_float(tf.range(grid_width))
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    return bbox_corners


def center_size_bbox_to_corners_bbox(centers, sizes):
    """Converts bbox center-size representation to corners representation.

    Args:
        centers: a tensor with shape [N, 2] representing bounding box centers
        sizes: a tensor with shape [N, 2] representing bounding boxes

    Returns:
        corners: tensor with shape [N, 4] representing bounding boxes in corners
                 representation
    """
    return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)
