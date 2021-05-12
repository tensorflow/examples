# EfficientDet
[![Paper](http://img.shields.io/badge/Paper-arXiv.1911.09070-B3181B?logo=arXiv)](https://arxiv.org/abs/1911.09070)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google/automl/blob/master/efficientdet/tutorial.ipynb)
[![TensorFlow Hub](https://img.shields.io/badge/TF%20Hub-Models-FF6F00?logo=tensorflow)](https://tfhub.dev/s?network-architecture=efficientdet)



[1] Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.
Arxiv link: https://arxiv.org/abs/1911.09070

Updates:

  - May10/2021: Added EfficientDet-lite checkpoints (by Yuqi and TFLite team)
  - Mar25/2021: Added [Det-AdvProp](https://arxiv.org/abs/2103.13886) model checkpoints ([see this page](./Det-AdvProp.md)).
  - Jul20/2020: Added keras/TF2 and new SOTA D7x: 55.1mAP with 153ms.
  - Apr22/2020: Sped up end-to-end latency: D0 has up to >200 FPS throughput on Tesla V100.
    * A great collaboration with [@fsx950223](https://github.com/fsx950223).
  - Apr1/2020: Updated results for test-dev and added EfficientDet-D7.
  - Mar26/2020: Fixed a few bugs and updated all checkpoints/results.
  - Mar24/2020: Added tutorial with visualization and coco eval.
  - Mar13/2020: Released the initial code and models.

**Quick start tutorial: [tutorial.ipynb](tutorial.ipynb)**

**Quick install dependencies: ```pip install -r requirements.txt```**

## 1. About EfficientDet Models

EfficientDets are a family of object detection models, which achieve state-of-the-art 55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors.


EfficientDets are developed based on the advanced backbone, a new BiFPN, and a new scaling technique:

<p align="center">
<img src="./g3doc/network.png" width="800" />
</p>

  * **Backbone**: we employ [EfficientNets](https://arxiv.org/abs/1905.11946) as our backbone networks.
  * **BiFPN**: we propose BiFPN, a bi-directional feature network enhanced with fast normalization, which enables easy and fast feature fusion.
  * **Scaling**: we use a single compound scaling factor to govern the depth, width, and resolution for all backbone, feature & prediction networks.

Our model family starts from EfficientDet-D0, which has comparable accuracy as [YOLOv3](https://arxiv.org/abs/1804.02767). Then we scale up this baseline model using our compound scaling method to obtain a list of detection models EfficientDet-D1 to D6, with different trade-offs between accuracy and model complexity.


<table border="0">
<tr>
    <td>
    <img src="./g3doc/flops.png" width="100%" />
    </td>
    <td>
    <img src="./g3doc/params.png", width="100%" />
    </td>
</tr>
</table>

** For simplicity, we compare the whole detectors here. For more comparison on FPN/NAS-FPN/BiFPN, please see Table 4 of our [paper](https://arxiv.org/abs/1911.09070).



## 2. Pretrained EfficientDet Checkpoints

We have provided a list of EfficientDet checkpoints and results as follows:

|       Model    | AP<sup>test</sup>    |  AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>S</sub>   |  AP<sub>M</sub>    |  AP<sub>L</sub>   |  AP<sup>val</sup> | | #params | #FLOPs |
|----------     |------ |------ |------ | -------- | ------| ------| ------ |------ |------ |  :------: |
|     EfficientDet-D0 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d0_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d0_coco_test-dev2017.txt))    | 34.6 | 53.0 | 37.1 | 12.4 | 39.0 | 52.7 | 34.3 |  | 3.9M | 2.54B  |
|     EfficientDet-D1 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d1.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d1.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d1_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d1_coco_test-dev2017.txt))    | 40.5 | 59.1 | 43.7 | 18.3 | 45.0 | 57.5 | 40.2 | | 6.6M | 6.10B |
|     EfficientDet-D2 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d2.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d2.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d2_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d2_coco_test-dev2017.txt))    | 43.9 | 62.7 | 47.6 | 22.9 | 48.1 | 59.5 | 43.5 | | 8.1M | 11.0B |
|     EfficientDet-D3 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d3.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d3.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d3_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d3_coco_test-dev2017.txt))    | 47.2 | 65.9 | 51.2 | 27.2 | 51.0 | 62.1 | 46.8 | | 12.0M | 24.9B |
|     EfficientDet-D4 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d4_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d4_coco_test-dev2017.txt))    | 49.7 | 68.4 | 53.9 | 30.7 | 53.2 | 63.2 | 49.3 |  | 20.7M | 55.2B |
|     EfficientDet-D5 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d5.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d5.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d5_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d5_coco_test-dev2017.txt))    | 51.5 | 70.5 | 56.1 | 33.9 | 54.7 | 64.1 | 51.2 |  | 33.7M | 130B |
|     EfficientDet-D6 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d6.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d6.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d6_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d6_coco_test-dev2017.txt))    | 52.6 | 71.5 | 57.2 | 34.9 | 56.0 | 65.4 | 52.1 | | 51.9M  |  226B  |
|     EfficientDet-D7 ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d7_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d7_coco_test-dev2017.txt))    | 53.7 | 72.4 | 58.4 | 35.8 | 57.0 | 66.3 | 53.4 | | 51.9M  |  325B  |
|     EfficientDet-D7x ([h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7x.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d7x.tar.gz), [val](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/val/d7x_coco_val.txt), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/testdev/d7x_coco_test-dev2017.txt))    | 55.1 | 74.3 | 59.9 | 37.2 | 57.9 | 68.0 | 54.4 | | 77.0M  |  410B  |

<sup><em>val</em> denotes validation results, <em>test-dev</em> denotes test-dev2017 results. AP<sup>val</sup> is for validation accuracy, all other AP results in the table are for COCO test-dev2017. All accuracy numbers are for single-model single-scale without ensemble or test-time augmentation.  EfficientDet-D0 to D6 are trained for 300 epochs and D7/D7x are trained for 600 epochs.</sup>

For more accurate and robust EfficientDet, please see [this page](./Det-AdvProp.md), which contains a list of models trained with Det-AdvProp + AutoAugment (AA) described in [this paper](https://arxiv.org/abs/2103.13886). The obatined model is not only more accurate on clean images, but also much more robust against various corruptions and domain shift.

In addition, the following table includes a list of models trained with fixed 640x640 image sizes (see appendix of [this paper](https://arxiv.org/abs/1911.09070)):


|       Model    |   mAP | Latency |
| ------ | ------ | ------  |
| D2(640) [h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d2-640.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d2-640.tar.gz) |  41.7 | 14.8ms |
| D3(640) [h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d3-640.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d3-640.tar.gz) |  44.0 | 18.7ms |
| D4(640) [h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d4-640.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d4-640.tar.gz) |  45.7 | 21.7ms |
| D5(640) [h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d5-640.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d5-640.tar.gz) |  46.6 | 26.6ms |
| D6(640) [h5](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d6-640.h5), [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco640/efficientdet-d6-640.tar.gz) |  47.9 | 33.8ms |

We have also provided a list of mobile-size lite models:

|       Model    |   mAP (float) | Quantized mAP (int8) | Prameters | Mobile latency |
| ------ | :------: | :------:  | :------: | :------:  |
| EfficientDet-lite0,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite0.tgz)  |  26.41 | 26.10 | 4.3M  |  36ms |
| EfficientDet-lite1,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite1.tgz)  |  31.50 | 31.12 |  5.8M |  49ms |
| EfficientDet-lite2,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite2.tgz)  |  35.06 | 34.69 |  7.2M | 69ms |
| EfficientDet-lite3,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite3.tgz)  |  38.77 | 38.42 |  11M  | 116ms |
| EfficientDet-lite3x, [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite3x..gz) |  42.64 | 41.87 |  12M  | 208ms  |
| EfficientDet-lite4,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite4.tgz)  |  43.18 | 42.83 |   20M | 260ms  |


## 3. Export SavedModel, frozen graph, tensort models, or tflite.


Run the following command line to export models:

    !rm  -rf savedmodeldir
    !python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 \
      --ckpt_path=efficientdet-d0 --saved_model_dir=savedmodeldir \
      --tensorrt=FP32  --tflite_path=efficientdet-d0.tflite \
      --hparams=voc_config.yaml

Then you will get:

 - saved model under `savedmodeldir/`
 - frozen graph with name `savedmodeldir/efficientdet-d0_frozen.pb`
 - TensorRT saved model under `savedmodeldir/tensorrt_fp32/`
 - tflite file with name `efficientdet-d0.tflite`

Notably,
 --tflite_path only works after 2.3.0-dev20200521 ,
 --ckpt_path=xx/archive is the folder for exporting the best model.


## 4. Benchmark model latency.


There are two types of latency: network latency and end-to-end latency.

(1) To measure the network latency (from the first conv to the last class/box
prediction output), use the following command:

    !python model_inspect.py --runmode=bm --model_name=efficientdet-d0

add --hparams="mixed_precision=True" if running on V100.

On single Tesla V100 without TensorRT, our D0 network (no pre/post-processing)
has 134 FPS (frame per second) for batch size 1, and 238 FPS for batch size 8.

(2) To measure the end-to-end latency (from the input image to the final rendered
new image, including: image preprocessing, network, postprocessing and NMS),
use the following command:

    !rm  -rf /tmp/benchmark/
    !python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 \
      --ckpt_path=efficientdet-d0 --saved_model_dir=/tmp/benchmark/ \
      --hparams=mixed_precision=true

    !python model_inspect.py --runmode=saved_model_benchmark \
      --saved_model_dir=/tmp/benchmark/efficientdet-d0_frozen.pb \
      --model_name=efficientdet-d0  --input_image=testdata/img1.jpg  \
      --output_image_dir=/tmp/

On single Tesla V100 without using TensorRT, our end-to-end
latency and throughput are:


|       Model    |   mAP | batch1 latency |  batch1 throughput |  batch8 throughput |
| ------ | ------ | ------  | ------ | ------ |
| EfficientDet-D0 |  34.6 | 10.2ms | 97 fps | 209 fps |
| EfficientDet-D1 |  40.5 | 13.5ms | 74 fps | 140 fps |
| EfficientDet-D2 |  43.0 | 17.7ms | 57 fps | 97 fps  |
| EfficientDet-D3 |  47.5 | 28.0ms | 36 fps | 58 fps  |
| EfficientDet-D4 |  49.7 | 42.8ms | 23 fps | 35 fps  |
| EfficientDet-D5 |  51.5 | 72.5ms | 14 fps | 18 fps  |
| EfficientDet-D6 |  52.6 | 92.8ms | 11 fps | - fps  |
| EfficientDet-D7 |  53.7 | 122ms  | 8.2 fps | - fps  |
| EfficientDet-D7x |  55.1 | 153ms  | 6.5 fps | - fps  |

** FPS means frames per second (or images/second).

## 5. Inference for images.

    # Step0: download model and testing image.
    !export MODEL=efficientdet-d0
    !export CKPT_PATH=efficientdet-d0
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/${MODEL}.tar.gz
    !wget https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png -O img.png
    !tar xf ${MODEL}.tar.gz

    # Step 1: export saved model.
    !python model_inspect.py --runmode=saved_model \
      --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 \
      --hparams="image_size=1920x1280" \
      --saved_model_dir=/tmp/saved_model

    # Step 2: do inference with saved model.
    !python model_inspect.py --runmode=saved_model_infer \
      --model_name=efficientdet-d0  \
      --saved_model_dir=/tmp/saved_model  \
      --input_image=img.png --output_image_dir=/tmp/
    # you can visualize the output /tmp/0.jpg


Alternatively, if you want to do inference using frozen graph instead of saved model, you can run

    # Step 0 and 1 is the same as before.
    # Step 2: do inference with frozen graph.
    !python model_inspect.py --runmode=saved_model_infer \
      --model_name=efficientdet-d0  \
      --saved_model_dir=/tmp/saved_model/efficientdet-d0_frozen.pb  \
      --input_image=img.png --output_image_dir=/tmp/

Lastly, if you only have one image and just want to run a quick test, you can also run the following command (it is slow because it needs to construct the graph from scratch):

    # Run inference for a single image.
    !python model_inspect.py --runmode=infer --model_name=$MODEL \
      --hparams="image_size=1920x1280"  --max_boxes_to_draw=100   --min_score_thresh=0.4 \
      --ckpt_path=$CKPT_PATH --input_image=img.png --output_image_dir=/tmp
    # you can visualize the output /tmp/0.jpg

Here is an example of EfficientDet-D0 visualization: more on [tutorial](tutorial.ipynb)

<p align="center">
<img src="./g3doc/street.jpg" width="800" />
</p>

## 6. Inference for videos.

You can run inference for a video and show the results online:

    # step 0: download the example video.
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/data/video480p.mov -O input.mov

    # step 1: export saved model.
    !python model_inspect.py --runmode=saved_model \
      --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 \
      --saved_model_dir=/tmp/savedmodel --hparams=voc_config.yaml

    # step 2: inference video using saved_model_video.
    !python model_inspect.py --runmode=saved_model_video \
      --model_name=efficientdet-d0 \
      --saved_model_dir=/tmp/savedmodel --input_video=input.mov

    # alternative step 2: inference video and save the result.
    !python model_inspect.py --runmode=saved_model_video \
      --model_name=efficientdet-d0   \
      --saved_model_dir=/tmp/savedmodel --input_video=input.mov  \
      --output_video=output.mov

## 7. Eval on COCO 2017 val or test-dev.

    // Download coco data.
    !wget http://images.cocodataset.org/zips/val2017.zip
    !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    !unzip val2017.zip
    !unzip annotations_trainval2017.zip

    // convert coco data to tfrecord.
    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
        --image_dir=val2017 \
        --caption_annotations_file=annotations/captions_val2017.json \
        --output_file_prefix=tfrecord/val \
        --num_shards=32

    // Run eval.
    !python main.py --mode=eval  \
        --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
        --val_file_pattern=tfrecord/val*  \
        --val_json_file=annotations/instances_val2017.json

You can also run eval on test-dev set with the following command:

    !wget http://images.cocodataset.org/zips/test2017.zip
    !unzip -q test2017.zip
    !wget http://images.cocodataset.org/annotations/image_info_test2017.zip
    !unzip image_info_test2017.zip

    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
          --image_dir=test2017 \
          --image_info_file=annotations/image_info_test-dev2017.json \
          --output_file_prefix=tfrecord/testdev \
          --num_shards=32

    # Eval on test-dev: testdev_dir must be set.
    # Also, test-dev has 20288 images rather than val 5000 images.
    !python main.py --mode=eval  \
        --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
        --val_file_pattern=tfrecord/testdev*  \
        --testdev_dir='testdev_output' --eval_samples=20288
    # Now you can submit testdev_output/detections_test-dev2017_test_results.json to
    # coco server: https://competitions.codalab.org/competitions/20794#participate

## 8. Finetune on PASCAL VOC 2012 with detector COCO ckpt.

Download data and checkpoints.

    # Download and convert pascal data.
    !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    !tar xf VOCtrainval_11-May-2012.tar
    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_pascal_tfrecord.py  \
        --data_dir=VOCdevkit --year=VOC2012  --output_path=tfrecord/pascal

    # Download backbone checkopints.
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz
    !tar xf efficientdet-d0.tar.gz

Create a config file for the PASCAL VOC dataset called voc_config.yaml and put this in it.

      num_classes: 21
      var_freeze_expr: '(efficientnet|fpn_cells|resample_p6)'
      label_map: {1: aeroplane, 2: bicycle, 3: bird, 4: boat, 5: bottle, 6: bus, 7: car, 8: cat, 9: chair, 10: cow, 11: diningtable, 12: dog, 13: horse, 14: motorbike, 15: person, 16: pottedplant, 17: sheep, 18: sofa, 19: train, 20: tvmonitor}

Finetune needs to use --ckpt rather than --backbone_ckpt.

    !python main.py --mode=train_and_eval \
        --train_file_pattern=tfrecord/pascal*.tfrecord \
        --val_file_pattern=tfrecord/pascal*.tfrecord \
        --model_name=efficientdet-d0 \
        --model_dir=/tmp/efficientdet-d0-finetune  \
        --ckpt=efficientdet-d0  \
        --train_batch_size=64 \
        --eval_batch_size=64 --eval_samples=1024 \
        --num_examples_per_epoch=5717 --num_epochs=50  \
        --hparams=voc_config.yaml

If you want to continue to train the model, simply re-run the above command because the `num_epochs` is a maximum number of epochs. For example, to reproduce the result of efficientdet-d0, set `--num_epochs=300` then run the command multiple times until the training is finished.

If you want to do inference for custom data, you can run

    # Setting hparams-flag is needed sometimes.
    !python model_inspect.py --runmode=infer \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --hparams=voc_config.yaml  \
      --input_image=img.png --output_image_dir=/tmp/

You should check more details of runmode which is written in caption-4.

## 9. Train on multi GPUs.

Create a config file for the PASCAL VOC dataset called voc_config.yaml and put this in it.

      num_classes: 21
      var_freeze_expr: '(efficientnet|fpn_cells|resample_p6)'
      label_map: {1: aeroplane, 2: bicycle, 3: bird, 4: boat, 5: bottle, 6: bus, 7: car, 8: cat, 9: chair, 10: cow, 11: diningtable, 12: dog, 13: horse, 14: motorbike, 15: person, 16: pottedplant, 17: sheep, 18: sofa, 19: train, 20: tvmonitor}

Download efficientdet coco checkpoint.

    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz
    !tar xf efficientdet-d0.tar.gz

Finetune needs to use --ckpt rather than --backbone_ckpt.

    python main.py --mode=train \
        --train_file_pattern=tfrecord/pascal*.tfrecord \
        --val_file_pattern=tfrecord/pascal*.tfrecord \
        --model_name=efficientdet-d0 \
        --model_dir=/tmp/efficientdet-d0-finetune  \
        --ckpt=efficientdet-d0  \
        --train_batch_size=64 \
        --eval_batch_size=64 --eval_samples=1024 \
        --num_examples_per_epoch=5717 --num_epochs=50  \
        --hparams=voc_config.yaml
        --strategy=gpus

If you want to do inference for custom data, you can run

    # Setting hparams-flag is needed sometimes.
    !python model_inspect.py --runmode=infer \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --hparams=voc_config.yaml  \
      --input_image=img.png --output_image_dir=/tmp/

You should check more details of runmode which is written in caption-4.

## 10. Training EfficientDets on TPUs.

To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource.
   * A GCS bucket to store your training checkpoints (the "model directory").
   * Install latest TensorFlow for both GCE VM and Cloud.

Then train the model:

    !export PYTHONPATH="$PYTHONPATH:/path/to/models"
    !python main.py --tpu=TPU_NAME --train_file_pattern=DATA_DIR/*.tfrecord --model_dir=MODEL_DIR --strategy=tpu

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access.
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.


For more instructions about training on TPUs, please refer to the following tutorials:

  * EfficientNet tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet

## 11. Reducing Memory Usage when Training EfficientDets on GPU.

EfficientDets use a lot of GPU memory for a few reasons:

* Large input resolution: because resolution is one of the scaling dimension, our resolution tends to be higher, which significantly increase activations (although no parameter increase).
* Large internal activations for backbone: our backbone uses a relatively large expansion ratio (6), causing the large expanded activations.
* Deep BiFPN: our BiFPN has multiple top-down and bottom-up paths, which leads to a lot of intermediate memory usage during training.

To train this model on GPU with low memory there is an experimental option grad_checkpoint.

Check these links for a high-level idea of what gradient checkpointing is doing:
1. https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9

**grad_checkpoint: True**

If set to True, keras model uses ```tf.recompute_grad``` to achieve gradient checkpoints.

Testing shows that:
* It also allows to train a d6 network with batch size of 2 by main.py on a 11Gb (1080Ti) GPU

## 12. Visualize TF-Records.

You can visualize tf-records with following commands:

To visualize training tfrecords with input dataloader use.
```
python dataset/inspect_tfrecords.py --file_pattern dataset/sample.record\ 
--model_name "efficientdet-d0" --samples 10\ 
--save_samples_dir train_samples/  -hparams="label_map={1:'label1'}, autoaugmentation_policy=v3"

```

To visualize evaluation tfrecords use.
```
python dataset/inspect_tfrecords.py --file_pattern dataset/sample.record\ 
--model_name "efficientdet-d0" --samples 10\ 
--save_samples_dir train_samples/  -eval\
-hparams="label_map={1:'label1'}"

```
* samples: random samples to visualize.
* model_name: model name will be used to get image_size.
* save_samples_dir: save dir.
* eval: flag for eval data.

NOTE: this is not an official Google product.
