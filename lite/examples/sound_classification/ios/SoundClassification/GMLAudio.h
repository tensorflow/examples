//
//  GMLAudio.h
//  SoundClassification
//
//  Created by Prianka Kariat on 19/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN



@interface GMLFloatBuffer : NSObject

@property (nonatomic, readonly) NSInteger length;
@property (nonatomic, readonly) float *buffer;


@end

@interface GMLAudioFormat : NSObject

@property (nonatomic, readonly) NSUInteger channelCount;
@property (nonatomic, readonly) NSUInteger sampleRate;

-(instancetype)initWithChannelCount:(NSInteger)channelCount sampleRate:(NSInteger)sampleRate;

@end

@interface GMLAudioRingBuffer : NSObject
- (instancetype)initWithBufferLength:(NSInteger)length;
-(void)loadWithNewData:(GMLFloatBuffer *)data offset:(NSInteger)offset length:(NSInteger)length;

@end


@interface GMLAudio : NSObject

@property (nonatomic, readonly) GMLAudioFormat *audioFormat;
@property (nonatomic, readonly) GMLFloatBuffer *buffer;

-(instancetype)initWithAudioFormat:(GMLAudioFormat *)format modelSampleCount:(NSInteger)modelSampleCount;

@end

NS_ASSUME_NONNULL_END
