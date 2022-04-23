// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "GMLUtils.h"
#import "GMLAudioError.h"

/** Error domain of GML Audio related errors. */
static NSString *const GMLAudioErrorDomain = @"org.gml.audio";

@implementation GMLUtils

+ (void)createCustomError:(NSError **)error
                 withCode:(NSInteger)code
              description:(NSString *)description {
  if (error) {
    *error = [NSError errorWithDomain:GMLAudioErrorDomain
                                 code:code
                             userInfo:@{NSLocalizedDescriptionKey : description}];
  }
}

@end
