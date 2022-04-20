//
//  GMLAudio.h
//  SoundClassification
//
//  Created by Prianka Kariat on 19/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface GMLAudioFormat : NSObject

@property (nonatomic) NSInteger channelCount;
@property (nonatomic) NSInteger sampleRate;


@end

@interface GMLFloatBuffer : NSObject

@property (nonatomic) float *buffer;
@property (nonatomic) NSInteger length;


@end

@interface GMLAudioFormat : NSObject

@property (nonatomic) NSInteger channelCount;
@property (nonatomic) NSInteger sampleRate;


@end

@interface GMLAudioRingBuffer : NSObject

@end



@interface GMLAudio : NSObject

@property (nonatomic, readonly) GMLAudioFormat *audioFormat;
@property (nonatomic, readonly) GMLFloatBuffer *buffer;


@end

NS_ASSUME_NONNULL_END
