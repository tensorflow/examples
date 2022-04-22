//
//  GMLRingBuffer.h
//  TFLAudioRecord
//
//  Created by Prianka Kariat on 22/04/22.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface GMLFloatBuffer : NSObject

@property (nonatomic, readonly) NSInteger size;
@property (nonatomic, readonly) float *data;

-(instancetype)initWithData:(float * _Nullable)data size:(NSInteger)size;
-(instancetype)initWithSize:(NSInteger)size;


@end

@interface GMLAudioFormat : NSObject

@property (nonatomic, readonly) NSUInteger channelCount;
@property (nonatomic, readonly) NSUInteger sampleRate;

-(instancetype)initWithChannelCount:(NSInteger)channelCount sampleRate:(NSInteger)sampleRate;

@end

@interface GMLAudioRingBuffer : NSObject
@property (nonatomic, readonly) GMLFloatBuffer *buffer;

- (instancetype)initWithBufferSize:(NSInteger)size;
-(void)loadWithNewBuffer:(GMLFloatBuffer *)data offset:(NSInteger)offset size:(NSInteger)length;

@end


NS_ASSUME_NONNULL_END
