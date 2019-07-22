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

ap = ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="Weights Directory")
ap.add_argument("-l", "--labels", rTileequired=True, help="labels")
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
	image = visualize.apply_mask(img, mask, color, alpha=0.5)

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

	cv2.imshow(image)