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

#ifndef JAVA_INTEROP_H_  // NOLINT(build/header_guard)
#define JAVA_INTEROP_H_

#include <jni.h>

#include <string>

#include "logging_assert.h"  // NOLINT (build/include)

namespace util {
namespace java {

class LocalRefDeleter {
 public:
  // Style guide violating implicit constructor so that the LocalRefDeleter
  // is implicitly constructed from the second argument to ScopedLocalRef.
  // TODO: remove NOLINT and the comment itself before publishing.
  LocalRefDeleter(JNIEnv* env) : env_(env) {}  // NOLINT(runtime/explicit)

  LocalRefDeleter(const LocalRefDeleter& orig) = default;

  // Copy assignment to allow move semantics in ScopedLocalRef.
  LocalRefDeleter& operator=(const LocalRefDeleter& rhs) { return *this; }

  void operator()(jobject o) const { env_->DeleteLocalRef(o); }

 private:
  JNIEnv* const env_;
};

template <typename T>
using ScopedLocalRef =
    std::unique_ptr<typename std::remove_pointer<T>::type, LocalRefDeleter>;

// Convenient wrappers that prevent from common JNI usage mistakes.
// We explicitly handle only the happy path of the JNI calls itself.
// The JNI error handling is out of scope for this sample app.

ScopedLocalRef<jclass> GetObjectClass(JNIEnv* env, jobject obj) {
  return ScopedLocalRef<jclass>(env->GetObjectClass(obj), env);
}

jmethodID GetMethodIdFromObject(JNIEnv* env, jobject obj, const char* name,
                                const char* sig) {
  auto clazz = GetObjectClass(env, obj);
  return env->GetMethodID(clazz.get(), name, sig);
}

ScopedLocalRef<jstring> NewStringUTF(JNIEnv* env, const char* mutf8_bytes) {
  return ScopedLocalRef<jstring>(env->NewStringUTF(mutf8_bytes), env);
}

std::string StringFromJString(JNIEnv* env, jstring value) {
  const char* data = env->GetStringUTFChars(value, nullptr);
  std::string copy(data);
  env->ReleaseStringUTFChars(value, data);
  return copy;
}

std::unique_ptr<jfloat[]> FloatArrayFromJFloatArray(JNIEnv* env,
                                                    jfloatArray array,
                                                    jsize* length_out_arg) {
  jint length = env->GetArrayLength(array);
  if (length_out_arg != nullptr) {
    *length_out_arg = length;
  }
  std::unique_ptr<jfloat[]> result(new jfloat[length]);
  env->GetFloatArrayRegion(array, 0, length, result.get());
  return result;
}

jfloatArray JfloatArrayFromFloatArray(JNIEnv* env, jfloat* array, jsize size) {
  jfloatArray result = env->NewFloatArray(size);
  env->SetFloatArrayRegion(result, 0, size, array);
  return result;
}

}  // namespace java
}  // namespace util
#endif  // JAVA_INTEROP_H_
