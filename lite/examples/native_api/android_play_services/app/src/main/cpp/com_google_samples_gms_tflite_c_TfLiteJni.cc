/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "flatbuffers/flatbuffers.h"
#include "java_interop.h"    // NOLINT (build/include)
#include "logging_assert.h"  // NOLINT (build/include)
#include "tensorflow/lite/abi/tflite.h"
#include "tensorflow/lite/acceleration/configuration/c/gpu_plugin.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/c_api.h"

namespace {
AAsset* g_model_asset = nullptr;
TfLiteInterpreter* g_interpreter = nullptr;
TfLiteOpaqueDelegate* g_gpu_delegate = nullptr;

TfLiteInterpreter* CreateInterpreterFromModel(JNIEnv* env, jobject callback,
                                              TfLiteModel* model);
TfLiteTensor* GetAndValidateInputTensor(JNIEnv* env, jobject tfliteJni,
                                        TfLiteInterpreter* interpreter,
                                        size_t input_size);
const TfLiteTensor* GetAndValidateOutputTensor(JNIEnv* env, jobject tfliteJni,
                                               TfLiteInterpreter* interpreter);

void LogToCallback(JNIEnv* env, jobject tfliteJni, const char* messageFormat,
                   ...) {
  static jmethodID printMethodId = util::java::GetMethodIdFromObject(
      env, tfliteJni, "sendLogMessage", "(Ljava/lang/String;)V");

  char message[1024];
  va_list args;
  va_start(args, messageFormat);
  vsnprintf(message, sizeof(message), messageFormat, args);
  va_end(args);

  auto jmessage = util::java::NewStringUTF(env, message);
  env->CallVoidMethod(tfliteJni, printMethodId, jmessage.get());
}

}  // namespace

extern "C" void
Java_com_google_samples_gms_tflite_c_TfLiteJni_initGpuAcceleration(
    JNIEnv* env, jobject tfliteJni) {
  if (!GmsTfLiteCheckInitializedOrThrow(env)) return;

  flatbuffers::FlatBufferBuilder fbb;
  tflite::TFLiteSettingsBuilder builder(fbb);
  const tflite::TFLiteSettings* tflite_settings =
      flatbuffers::GetTemporaryPointer(fbb, builder.Finish());
  ASSERT_NE(env, tflite_settings, nullptr);

  const TfLiteOpaqueDelegatePlugin* pluginCApi = TfLiteGpuDelegatePluginCApi();
  ASSERT_NE(env, pluginCApi, nullptr);
  g_gpu_delegate = pluginCApi->create(tflite_settings);
  ASSERT_NE(env, g_gpu_delegate, nullptr);
  LogToCallback(env, tfliteJni, "GPU acceleration initialized");
}

extern "C" void Java_com_google_samples_gms_tflite_c_TfLiteJni_loadModel(
    JNIEnv* env, jobject tfliteJni, jobject asset_manager, jstring asset_name) {
  if (!GmsTfLiteCheckInitializedOrThrow(env)) return;
  // Create model.
  AAssetManager* aAssetManager = AAssetManager_fromJava(env, asset_manager);
  g_model_asset = AAssetManager_open(
      aAssetManager, util::java::StringFromJString(env, asset_name).c_str(),
      AASSET_MODE_BUFFER);
  TfLiteModel* model = TfLiteModelCreate(AAsset_getBuffer(g_model_asset),
                                         AAsset_getLength(g_model_asset));
  ASSERT_NE(env, model, nullptr);
  LogToCallback(env, tfliteJni, "  Model created");

  g_interpreter = CreateInterpreterFromModel(env, tfliteJni, model);

  ASSERT_EQ(env, TfLiteInterpreterAllocateTensors(g_interpreter), kTfLiteOk);
}

extern "C" jfloatArray
Java_com_google_samples_gms_tflite_c_TfLiteJni_runInference(JNIEnv* env,
                                                            jobject tfliteJni,
                                                            jfloatArray input) {
  jsize input_size;
  auto cinput = util::java::FloatArrayFromJFloatArray(env, input, &input_size);

  // Copy input to g_interpreter.
  TfLiteTensor* input_tensor =
      GetAndValidateInputTensor(env, tfliteJni, g_interpreter, input_size);
  ASSERT_EQ(env,
            TfLiteTensorCopyFromBuffer(input_tensor, cinput.get(),
                                       input_size * sizeof(float)),
            kTfLiteOk);
  LogToCallback(env, tfliteJni, "  Input copied");

  // Run inference.
  ASSERT_EQ(env, TfLiteInterpreterInvoke(g_interpreter), kTfLiteOk);
  LogToCallback(env, tfliteJni, "  Inference executed");

  // Get output from g_interpreter.
  const TfLiteTensor* output_tensor =
      GetAndValidateOutputTensor(env, tfliteJni, g_interpreter);
  float c_array[2];
  ASSERT_EQ(env,
            TfLiteTensorCopyToBuffer(output_tensor, c_array, 2 * sizeof(float)),
            kTfLiteOk);
  LogToCallback(env, tfliteJni, "  Output values copied");
  return util::java::JfloatArrayFromFloatArray(env, c_array, 2);
}

extern "C" void Java_com_google_samples_gms_tflite_c_TfLiteJni_destroy(
    JNIEnv* env, jobject tfliteJni) {
  TfLiteInterpreterDelete(g_interpreter);
  g_interpreter = nullptr;
  // Some of the resources in the g_interpreter, e.g. the tensor names, are
  // backed by the g_model_asset storage, so we should not close the
  // g_model_asset until after we're done with the g_interpreter.
  AAsset_close(g_model_asset);
  g_model_asset = nullptr;
  if (g_gpu_delegate) {
    const auto pluginCApi = TfLiteGpuDelegatePluginCApi();
    ASSERT_NE(env, pluginCApi, nullptr);
    pluginCApi->destroy(g_gpu_delegate);
    g_gpu_delegate = nullptr;
  }
}

namespace {

TfLiteInterpreter* CreateInterpreterFromModel(JNIEnv* env, jobject callback,
                                              TfLiteModel* model) {
  // Set up options
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(env, options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  if (g_gpu_delegate) {
    TfLiteInterpreterOptionsAddDelegate(options, g_gpu_delegate);
    LogToCallback(env, callback, "  GPU delegate added to options");
  }
  LogToCallback(env, callback, "  Options created");

  // Create interpreter.
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(env, interpreter, nullptr);
  LogToCallback(env, callback, "  Interpreter created");

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  LogToCallback(env, callback, "  Options/model closed/deleted");
  return interpreter;
}

TfLiteTensor* GetAndValidateInputTensor(JNIEnv* env, jobject tfliteJni,
                                        TfLiteInterpreter* interpreter,
                                        size_t input_size) {
  ASSERT_EQ(env, TfLiteInterpreterGetInputTensorCount(interpreter), 1);

  int input_dims[1] = {static_cast<int>(input_size)};
  ASSERT_EQ(env,
            TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 1),
            kTfLiteOk);
  ASSERT_EQ(env, TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  LogToCallback(env, tfliteJni, "  Input dims checked");

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(env, input_tensor, nullptr);
  ASSERT_EQ(env, TfLiteTensorType(input_tensor), kTfLiteFloat32);
  ASSERT_EQ(env, TfLiteTensorNumDims(input_tensor), 1);
  ASSERT_EQ(env, TfLiteTensorDim(input_tensor, 0), input_size);
  ASSERT_EQ(env, TfLiteTensorByteSize(input_tensor),
            sizeof(float) * input_size);
  ASSERT_NE(env, TfLiteTensorData(input_tensor), nullptr);
  ASSERT_STREQ(env, TfLiteTensorName(input_tensor), "input");
  LogToCallback(env, tfliteJni, "  Input tensor params checked");

  auto input_params = TfLiteTensorQuantizationParams(input_tensor);
  ASSERT_EQ(env, input_params.scale, 0.f);
  ASSERT_EQ(env, input_params.zero_point, 0);
  LogToCallback(env, tfliteJni, "  Input quantization params checked");

  return input_tensor;
}

const TfLiteTensor* GetAndValidateOutputTensor(JNIEnv* env, jobject tfliteJni,
                                               TfLiteInterpreter* interpreter) {
  ASSERT_EQ(env, TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(env, output_tensor, nullptr);
  ASSERT_EQ(env, TfLiteTensorType(output_tensor), kTfLiteFloat32);
  ASSERT_EQ(env, TfLiteTensorNumDims(output_tensor), 1);
  ASSERT_EQ(env, TfLiteTensorDim(output_tensor, 0), 2);
  ASSERT_EQ(env, TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
  ASSERT_NE(env, TfLiteTensorData(output_tensor), nullptr);
  ASSERT_STREQ(env, TfLiteTensorName(output_tensor), "output");
  LogToCallback(env, tfliteJni, "  Output tensor params checked");

  auto output_params = TfLiteTensorQuantizationParams(output_tensor);
  ASSERT_EQ(env, output_params.scale, 0.f);
  ASSERT_EQ(env, output_params.zero_point, 0);
  LogToCallback(env, tfliteJni, "  Output quantization params checked");

  return output_tensor;
}

}  // namespace
