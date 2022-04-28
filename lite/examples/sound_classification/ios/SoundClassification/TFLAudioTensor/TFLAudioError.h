// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLAudioErrorCode
 * This enum specifies  error codes for audio TensorFlow Lite Task Library.
 */
typedef NS_ENUM(NSUInteger, TFLAudioErrorCode) {

  /** Unspecified error. */
  TFLAudioErrorCodeUnspecifiedError = 1,

  /** Invalid argument specified. */
  TFLAudioErrorCodeInvalidArgumentError = 2,

  TFLAudioErrorCodeAudioProcessingError = 3,

  /** Record permissions denied by user*/
  TFLAudioErrorCodeRecordPermissionDeniedError = 4,

  /** Record undetermined. Permissions have to be requested uusing AVAudioSession.*/
  TFLAudioErrorCodeRecordPermissionUndeterminedError = 4,

  TFLAudioErrorCodeWaitingForNewInputError = 5,

  /** kInternal indicates an internal error has occurred and some invariants expected by the
   * underlying system have not been satisfied. This error code is reserved for serious errors.
   */
  TFLAudioErrorCodeInternalError

} NS_SWIFT_NAME(AudioErrorCode);

NS_ASSUME_NONNULL_END
