from enum import Enum
import tensorflow as tf
import numpy as np
import math


class BodyPart(Enum):
    """Class for body part encoding, source - https://www.tensorflow.org/lite/models/pose_estimation/overview """
    nose = 0
    leftEye = 1
    rightEye = 2
    leftEar = 3
    rightEar = 4
    leftShoulder = 5
    rightShoulder = 6
    leftElbow = 7
    rightElbow = 8
    leftWrist = 9
    rightWrist = 10
    leftHip = 11
    rightHip = 12
    leftKnee = 13
    rightKnee = 14
    leftAnkle = 15
    rightAnkle = 16


class Posenet:
    """
    Idea source - https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android
    Load, init net for python - https://www.tensorflow.org/lite/guide/inference
    """
    # Source image shape
    source_shape = None
    # Net input shape
    net_inp_shape = None

    def __init__(self, net_path="model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"):
        """
        Load model from path. After firts run you must load form
        https://www.tensorflow.org/lite/models/pose_estimation/overview
        :param net_path: path for model
        """
        self.interpreter = tf.lite.Interpreter(model_path=net_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __normalize(self, img_array):
        """
        Image resize
        :param img_array: np_array with RGB image
        :return: resize np_array
        """
        self.source_shape = img_array.shape
        in_image = tf.cast(img_array, tf.float32) / 128.0 - 1
        np_frame = np.expand_dims(in_image, axis=0)
        input_shape = self.input_details[0]['shape']
        self.net_inp_shape = input_shape
        return tf.image.resize(np_frame, (input_shape[1], input_shape[2]))

    @staticmethod
    def __sigmoid(x):
        """It is magic"""
        return 1/(1 + math.exp(-x))

    @staticmethod
    def __body_part_raw_columns(array_3d):
        """
        Transform model output (np_array) to dict with body part name and one point
        :param array_3d: np_array
        :return: dict like {'nose': (170, 82), 'leftEye': (171, 78) ...}
        """
        shape = array_3d.shape
        key_point_positions = dict()
        for part in BodyPart:
            buf = array_3d[:, :, part.value]
            max_val = buf[0, 0]
            max_row = 0
            max_col = 0
            for row in range(0, shape[0]):
                for col in range(0, shape[1]):
                    s = Posenet.__sigmoid(buf[row, col])
                    if s > max_val:
                        max_val = s
                        max_row = row
                        max_col = col
            key_point_positions[part.name] = (max_row, max_col)
        return key_point_positions

    def pose(self, input_image):
        """
        Main class, run it after unit
        :param input_image: np_array with RGB image
        :return: dict with body path like {'nose': (170, 82), 'leftEye': (171, 78) ...}
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], self.__normalize(input_image))
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        part_in_net = Posenet.__body_part_raw_columns(output_data[0, :, :, :])
        return self.__calculating(part_in_net)

    def __calculating(self, parts):
        """
        Resize and offset points for body part. Make change x and y
        :param parts: dict with body part name as key and poit (x,y) as value
        :return: dict like parts, but resize
        """
        kx = self.net_inp_shape[2] / (self.output_details[0]['shape'][2] - 1)
        ky = self.net_inp_shape[1] / (self.output_details[0]['shape'][1] - 1)
        k2y = self.source_shape[0] / self.net_inp_shape[1]
        k2x = self.source_shape[1] / self.net_inp_shape[2]
        offset = self.interpreter.get_tensor(self.output_details[1]['index'])
        for key in parts.keys():
            # Change x and y
            p = parts[key]
            b_index = BodyPart[key].value
            y = (p[0] * ky + offset[0][p[0]][p[1]][b_index]) * k2y
            x = (p[1] * kx + offset[0][p[0]][p[1]][b_index + self.output_details[0]['shape'][3]]) * k2x
            parts[key] = (int(x), int(y))
        return parts
