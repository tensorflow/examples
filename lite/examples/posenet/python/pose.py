import posenet
import os.path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tensorflow as tf
import glob


def draw_line(l, s, color):
    """
    Draw line
    :param l: list body part name for lines
    :param s: dict with all body part and points
    :param color: color for line
    :return: list matplot line2d
    """
    line_list = list()
    i = 1
    while i < len(l):
        line_list.append(mlines.Line2D((s[l[i - 1]][0], s[l[i]][0]), (s[l[i - 1]][1], s[l[i]][1]),
                                   linewidth=3, color=color))
        i += 1
    return line_list


def display(source_img, skelet, file_path):
    """
    Save image with bone and points
    :param source_img: source image
    :param skelet: dict with all body part and points
    :param file_path: full name for file save
    :return: none, result save in file
    """
    plt.figure(figsize=(15, 15))
    for i in range(len(source_img)):
        plt.subplot(1, len(source_img), i + 1)
        plt.title("Source image with bone")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(source_img[i]))
        for part in skelet:
            circle = plt.Circle(skelet[part], 4, color='r', fill=True)
            plt.gcf().gca().add_artist(circle)
        lines = list()
        # Draw left body path
        lines += draw_line(('leftWrist', 'leftElbow', 'leftShoulder', 'leftHip', 'leftKnee', 'leftAnkle'),
                           skelet, "green")
        # Draw right body path
        lines += draw_line(('rightWrist', 'rightElbow', 'rightShoulder', 'rightHip', 'rightKnee', 'rightAnkle'),
                           skelet, "red")
        # Draw middle path
        lines += draw_line(('leftHip', 'rightHip'), skelet, "blue")
        lines += draw_line(('leftShoulder', 'rightShoulder'), skelet, "blue")
        for l in lines:
            plt.gca().add_line(l)
        plt.axis('off')
    plt.savefig(file_path)
    plt.close()


if __name__ == "__main__":
    SOURCE = "images/source"
    OUTPUT = "images/output"
    # Check model path!
    posenet = posenet.Posenet("model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")

    for f_name in glob.glob(os.path.join(SOURCE, "*.jpg")):
        print("Processing file:", f_name)
        input_image = plt.imread(f_name)
        result = posenet.pose(input_image)
        display((input_image,), result, os.path.join(OUTPUT, os.path.split(f_name)[-1]))
        print(result)
        print("=================\n")
