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

#ifndef LOGGING_ASSERT_H_  // NOLINT(build/header_guard)
#define LOGGING_ASSERT_H_

#include <stdio.h>

// We use an extra level of macro indirection here to ensure that the
// macro arguments get evaluated, so that in a call to CHECK_CONDITION(foo),
// the call to STRINGIZE(condition) in the definition of the CHECK_CONDITION
// macro results in the string "foo" rather than the string "condition".
#define STRINGIZE(expression) STRINGIZE2(expression)
#define STRINGIZE2(expression) #expression

// Will log the result to Android's logcat.
#define CHECK_CONDITION(jni_env, condition)                            \
  do {                                                                 \
    ((condition)                                                       \
     ? logging::CheckSucceeded(STRINGIZE(condition), __func__)         \
     : logging::CheckFailed(jni_env, STRINGIZE(condition), __func__)); \
  } while (false)
#define ASSERT_EQ(jni_env, expected, actual) \
  CHECK_CONDITION(jni_env, (expected) == (actual))
#define ASSERT_NE(jni_env, expected, actual) \
  CHECK_CONDITION(jni_env, (expected) != (actual))
#define ASSERT_STREQ(jni_env, expected, actual) \
  ASSERT_EQ(jni_env, 0, strcmp((expected), (actual)))

namespace logging {

void CheckSucceeded(const char* expression, const char* function) {
  char message[1024];
  snprintf(message, sizeof(message), "SUCCESS: CHECK passed: %s: %s", function,
           expression);
  __android_log_print(ANDROID_LOG_INFO, "TFLJNI", "%s\n", message);
}

void CheckFailed(JNIEnv* env, const char* expression, const char* function) {
  char message[1024];
  snprintf(message, sizeof(message), "ERROR: CHECK failed: %s: %s", function,
           expression);
  __android_log_print(ANDROID_LOG_ERROR, "TFLJNI", "%s\n", message);
  env->FatalError(message);
  // no return
  abort();
}
}  // namespace logging

#endif  // LOGGING_ASSERT_H_
