// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "ios/ImageClassifierBuilder/NSString+AbseilStringView.h"

@implementation NSString (AbseilStringView)

+ (instancetype)cbr_stringWithStringView:(const absl::string_view &)stringView {
  // Printing a string_view requires some adjustments. See tip at the bottom of
  // https://abseil.io/tips/1
  return [self stringWithFormat:@"%.*s", static_cast<int>(stringView.size()), stringView.data()];
}

@end
