# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mask R-CNN with TF 2.0 - Runner.

Reference:
[Mask R-CNN](https://arxiv.org/asb/1703.06870)
"""

from argparse import ArgumentParser
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import colorsys
import numpy as np
import random
from config import Config
import model as mask_rcnn
import visualize
import utils
import os
import cv2
import imutils
import pdb

ap = ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="Weights Directory")
ap.add_argument("-l", "--labels", required=True, help="labels")
ap.add_argument("-i", "--image", required=True, help="Images")
args = vars(ap.parse_args())

LABELS = open(args["labels"]).read().strip().split("\n")
file_name = "images/intersection.jpg"

class MaskConfig(Config):
	NAME = "COCO Trial 2"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = len(LABELS)

configs = MaskConfig()

# img = Image.open(args["image"]).convert("RGBA")
# img.resize((512, 512))
image = cv2.imread(args["image"])
image = imutils.resize(image, width=512)
hsv = [(i / len(LABELS), 1, 1.0) for i in range(len(LABELS))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

print("Loading model weights......")
model = mask_rcnn.MaskRCNN(mode="inference", config=configs, model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

print("Making inferences with Mask R-CNN......")
val = model.detect([image], verbose=1)[0]


for i in range(0, val["rois"].shape[0]):
	classID = val["class_ids"][i]
	mask = val["masks"][:, :, i]
	color = COLORS[classID][::-1]
	image = visualize.apply_mask(image, mask, color, alpha=0.5)

for i in range(0, len(val["scores"])):
	(startY, startX, endY, endX) = val["rois"][i]
	classID = val["class_ids"][i]
	label = LABELS[classID]
	score = val["scores"][i]
	color = [int(c) for c in np.array(COLORS[classID]) * 255]

	# draw = ImageDraw.Draw(img)
	# draw.rectangle(((startX, startY), (endX, endY)), fill="black")
	# draw.text((20, 70), f'{label}, {score}', font=ImageFont.truetype("font_path123"))
	#img.save("/content/", "JPEG")

	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	text = "{}: {:.3f}".format(label, score)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

	cv2.imshow('Mask R-CNN Prediction', image)
	cv2.waitKey(0)