//
//  GMLAudio.h
//  SoundClassification
//
//  Created by Prianka Kariat on 19/04/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "TFLAudioRecord.h"
#import "GMLRingBuffer.h"

NS_ASSUME_NONNULL_BEGIN


@interface GMLAudio : NSObject

@property (nonatomic, readonly) GMLAudioFormat *audioFormat;
@property (nonatomic, readonly) GMLAudioRingBuffer *buffer;

-(instancetype)initWithAudioFormat:(GMLAudioFormat *)format modelSampleCount:(NSInteger)modelSampleCount;
-(void)loadWithBuffer:(GMLFloatBuffer *)floatBuffer;


@end

NS_ASSUME_NONNULL_END
