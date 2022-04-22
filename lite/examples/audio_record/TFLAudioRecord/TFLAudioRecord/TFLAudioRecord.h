//
//  TFLAudioRecord.h
//  SoundClassification
//
//  Created by Prianka Kariat on 21/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "GMLRingBuffer.h"

@import AVFoundation;

NS_ASSUME_NONNULL_BEGIN

@interface TFLAudioRecord : NSObject

@property(nonatomic, readonly)GMLAudioFormat *audioFormat;
@property(nonatomic, readonly)NSInteger bufferSize;


//- (void)checkPermissionAndStartTappingMicrophoneWithCompletionHandler:(void(^)(NSError * _Nullable error))completionHandler;
- (void)checkAndStartTappingMicrophoneWithCompletionHandler:(void(^)(GMLFloatBuffer *_Nullable buffer, NSError * _Nullable error))completionHandler;

-(void)stopTappingMicrophone;

-(instancetype)initWithChannelCount:(NSInteger)channelCount sampleRate:(NSInteger)sampleRate bufferSize:(NSInteger)bufferSize;

//-(GMLFloatBuffer *)readWithData:(GMLFloatBuffer *)buffer offset:(NSInteger)offset length:(NSInteger)length;



@end

NS_ASSUME_NONNULL_END
