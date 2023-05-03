# Generative AI

## Introduction
Large language models (LLMs) are types of machine learning models that are created based on large bodies of text data to generate various outputs for natural language processing (NLP) tasks, including text generation, question answering, and machine translation. They are based on Transformer architecture and are trained on massive amounts of text data, often involving billions of words. Even LLMs of a smaller scale, such as GPT-2, can perform impressively. Converting TensorFlow models to a lighter, faster, and low-power model allows for us to run generative AI models on-device, with benefits of better user security because data will never leave your device.

 This example shows you how to build an Android app with TensorFlow Lite to run a Keras  LLM  and provides suggestions for model optimization using quantizing techniques l, which otherwise would require a much larger amount of memory and greater computational power to run.



## Guides
### Step 1. Train a language model using Keras

For this demonstration, we will use KerasNLP to get the GPT-2 model. KerasNLP is a library that contains state-of-the-art pretrained models for natural language processing tasks, and can support users through their entire development cycle. You can see the list of models available in the [KerasNLP repository](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models). The workflows are built from modular components that have state-of-the-art preset weights and architectures when used out-of-the-box and are easily customizable when more control is needed. Creating the GPT-2 model can be done with the following steps:

```python
gpt2_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")

gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
  "gpt2_base_en",
  sequence_length=256,
  add_end_token=True,
)

gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
  "gpt2_base_en", 
  preprocessor=gpt2_preprocessor,
)
```

You can check out the full GPT-2 model implementation [on GitHub](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models/gpt2).


### Step 2. Convert a Keras model to a TFLite model

Start with the `generate()` function from GPT2CausalLM that performs the conversion. Wrap the `generate()` function to create a concrete TensorFlow function:

```python
@tf.function
def generate(prompt, max_length):
  # prompt: input prompt to the LLM in string format
  # max_length: the max length of the generated tokens 
  return gpt2_lm.generate(prompt, max_length)
concrete_func = generate.get_concrete_function(tf.TensorSpec([], tf.string), 100)
```

Now define a helper function that will run inference with an input and a TFLite model. TensorFlow text ops are not built-in ops in the TFLite runtime, so you will need to add these custom ops in order for the interpreter to make inference on this model. This helper function accepts an input and a function that performs the conversion, namely the `generator()` function defined above. 

```python
def run_inference(input, generate_tflite):
  interp = interpreter.InterpreterWithCustomOps(
    model_content=generate_tflite,
    custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
  interp.get_signature_list()

  generator = interp.get_signature_runner('serving_default')
  output = generator(prompt=np.array([input]))
```

You can convert the model now:

```python
gpt2_lm.jit_compile = False
converter = tf.lite.TFLiteConverter.from_concrete_functions(
  [concrete_func],
  gpt2_lm)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite ops
  tf.lite.OpsSet.SELECT_TF_OPS, # enable TF ops
]
converter.allow_custom_ops = True
converter.target_spec.experimental_select_user_tf_ops = [
  "UnsortedSegmentJoin",
  "UpperBound"
]
converter._experimental_guarantee_all_funcs_one_use = True
generate_tflite = converter.convert()
run_inference("I'm enjoying a", generate_tflite)
```

### Step 3. Quantization
TensorFlow Lite has implemented an optimization technique called quantization which can  reduce model size and accelerate inference. Through the quantization process, 32-bit floats are mapped to smaller 8-bit integers, therefore reducing the model size by a factor of 4 for more efficient execution on modern hardwares. There are several ways to do quantization in TensorFlow. You can visit the [TFLite Model optimization](https://www.tensorflow.org/lite/performance/model_optimization) and [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) pages for more information. The types of quantizations are explained briefly below.

Here, you will use the post-training dynamic range quantization on the GPT-2 model by setting the converter optimization flag to tf.lite.Optimize.DEFAULT, and the rest of the conversion process is the same as detailed before. We tested that with this quantization technique the latency is around 6.7 seconds on Pixel 7 with max output length set to 100.

```python
gpt2_lm.jit_compile = False
converter = tf.lite.TFLiteConverter.from_concrete_functions(
  [concrete_func],
  gpt2_lm)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite ops
  tf.lite.OpsSet.SELECT_TF_OPS, # enable TF ops
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.experimental_select_user_tf_ops = [
  "UnsortedSegmentJoin",
  "UpperBound"
]
converter._experimental_guarantee_all_funcs_one_use = True
quant_generate_tflite = converter.convert()
run_inference("I'm enjoying a", quant_generate_tflite)

with open('quantized_gpt2.tflite', 'wb') as f:
  f.write(quant_generate_tflite)
```



### Step 4. Android App integration

You can clone this repo and substitute `android/app/src/main/assets/autocomplete.tflite` with your converted `quant_generate_tflite` file. Please refer to [how-to-build.md](https://github.com/tensorflow/examples/blob/master/lite/examples/generative_ai/android/how-to-build.md) to build this Android App. 

## Safety and Responsible AI
As noted in the original [OpenAI GPT-2 announcement](https://openai.com/research/better-language-models), there are [notable caveats and limitations](https://github.com/openai/gpt-2#some-caveats) with the GPT-2 model. In fact, LLMs today generally have some well-known challenges such as hallucinations, fairness, and bias; this is because these models are trained on real-world data, which make them reflect real world issues.
This codelab is created only to demonstrate how to create an app powered by LLMs with TensorFlow tooling. The model produced in this codelab is for educational purposes only and not intended for production usage.
LLM production usage requires thoughtful selection of training datasets and comprehensive safety mitigations. One such functionality offered in this Android app is the profanity filter, which rejects bad user inputs or model outputs. If any inappropriate language is detected, the app will in return reject that action. To learn more about Responsible AI in the context of LLMs, make sure to watch the Safe and Responsible Development with Generative Language Models technical session at Google I/O 2023 and check out the [Responsible AI Toolkit](https://www.tensorflow.org/responsible_ai).