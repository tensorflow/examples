/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <android/log.h>
#include <jni.h>

#include <cinttypes>
#include <cstring>
#include <string>

#include "SuperResolution.h"

namespace tflite {
namespace examples {
namespace superresolution {

extern "C" JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_superResolutionFromJNI(
    JNIEnv *env, jobject thiz, jlong native_handle, jintArray low_res_rgb) {
  jint *lr_img_rgb = env->GetIntArrayElements(low_res_rgb, NULL);

  if (!reinterpret_cast<SuperResolution *>(native_handle)
           ->IsInterpreterCreated()) {
    return nullptr;
  }

  // Generate super resolution image
  auto sr_rgb_colors = reinterpret_cast<SuperResolution *>(native_handle)
                           ->DoSuperResolution(static_cast<int *>(lr_img_rgb));
  if (!sr_rgb_colors) {
    return nullptr;  // super resolution failed
  }
  jintArray sr_img_rgb = env->NewIntArray(kNumberOfOutputPixels);
  env->SetIntArrayRegion(sr_img_rgb, 0, kNumberOfOutputPixels,
                         sr_rgb_colors.get());

  // Clean up before we return
  env->ReleaseIntArrayElements(low_res_rgb, lr_img_rgb, JNI_COMMIT);

  return sr_img_rgb;
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_initWithByteBufferFromJNI(
    JNIEnv *env, jobject thiz, jobject model_buffer, jboolean use_gpu) {
  const void *model_data =
      static_cast<void *>(env->GetDirectBufferAddress(model_buffer));
  jlong model_size_bytes = env->GetDirectBufferCapacity(model_buffer);
  SuperResolution *super_resolution = new SuperResolution(
      model_data, static_cast<size_t>(model_size_bytes), use_gpu);
  if (super_resolution->IsInterpreterCreated()) {
    LOGI("Interpreter is created successfully");
    return reinterpret_cast<jlong>(super_resolution);
  } else {
    delete super_resolution;
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_deinitFromJNI(
    JNIEnv *env, jobject thiz, jlong native_handle) {
  delete reinterpret_cast<SuperResolution*>(native_handle);
}

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite
