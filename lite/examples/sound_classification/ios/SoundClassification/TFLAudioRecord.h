//
//  TFLAudioRecord.h
//  SoundClassification
//
//  Created by Prianka Kariat on 21/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "GMLAudio.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLAudioRecord : NSObject

@property(nonatomic, readonly)GMLAudioFormat *audioFormat;
@property(nonatomic, readonly)NSUInteger bufferSize;


- (void)checkPermissionAndStartTappingMicrophoneWithCompletionHandler:(void(^)(NSError * _Nullable error))completionHandler;

-(instancetype)initWithChannelCount:(NSInteger)channelCount sampleRate:(NSInteger)sampleRate bufferSize:(NSInteger)bufferSize;



@end

NS_ASSUME_NONNULL_END
