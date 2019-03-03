#!/usr/bin/env bash

tflite_convert --output_file converted_speech_model.tflite \
--graph_def_file model.h5.pb \
--output_format TFLITE \
--inference_type FLOAT \
--inference_input_type FLOAT \
--input_arrays input_1 \
--output_arrays output_node0 \
#--allow_custom_ops
